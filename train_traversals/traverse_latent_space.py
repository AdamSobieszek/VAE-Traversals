import argparse
import os
import os.path as osp
import torch
from torch import nn
from PIL import Image, ImageDraw
import json
from torchvision.transforms import ToPILImage
from lib import *
from models.gan_load import (
    build_biggan,
    build_proggan,
    build_sngan,
    build_stylegan2_early_output,
    build_stylegan2mps,
)
from lib.utils import choose_device, sample_z


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def tensor2image(tensor, img_size=None, adaptive=False):
    # Squeeze tensor image
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def build_gan(exp_args):
    gan_type = exp_args.__dict__['gan_type']
    # -- BigGAN
    if gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                         target_classes=exp_args.__dict__.get('biggan_target_classes', None))
    # -- ProgGAN
    elif gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]])
    # -- StyleGAN2
    elif gan_type == 'StyleGAN2':
        early_output = exp_args.__dict__.get('early_output', False)
        compile_enabled = exp_args.__dict__.get('compile', False)
        stylegan2_resolution = exp_args.__dict__.get('stylegan2_resolution', 1024)
        weight_key = 'early_output' if early_output else stylegan2_resolution
        stylegan_builder_kwargs = dict(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][weight_key],
            shift_in_w_space=exp_args.__dict__.get('shift_in_w_space', False),
            compile=compile_enabled,
            compile_mode=exp_args.__dict__.get('compile_mode', 'default'),
            use_optimized=compile_enabled and (
                early_output or not exp_args.__dict__.get('no_optimized_synthesis', False)
            ),
            mixed_precision=exp_args.__dict__.get('mixed_precision', 'bf16'),
        )
        if early_output:
            G = build_stylegan2_early_output(**stylegan_builder_kwargs)
        else:
            G = build_stylegan2mps(
                resolution=stylegan2_resolution,
                **stylegan_builder_kwargs,
            )
    # -- Spectrally Normalised GAN (SNGAN)
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                        gan_type=gan_type)

    return G


def load_traversal_sets(exp_models_dir, device):
    """Load experiment arguments and the preferred traversal checkpoint."""
    args_json_file = osp.join(osp.dirname(exp_models_dir), 'args.json')
    if not osp.isfile(args_json_file):
        raise FileNotFoundError("File not found: {}".format(args_json_file))
    exp_args = ModelArgs(**json.load(open(args_json_file)))

    checkpoint_path = osp.join(exp_models_dir, 'checkpoint.pt')
    if not osp.isfile(checkpoint_path):
        candidates = []
        if osp.isfile(osp.join(exp_models_dir, 'traversal_sets.pt')):
            candidates.append('traversal_sets.pt')
        candidates.extend(sorted(
            filename for filename in os.listdir(exp_models_dir)
            if filename.startswith('traversal_sets-')
        ))
        if not candidates:
            raise FileNotFoundError("No traversal checkpoint found in {}".format(exp_models_dir))
        checkpoint_path = osp.join(exp_models_dir, candidates[-1])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return exp_args, checkpoint, checkpoint_path


def load_traversal_weights(traversal_pde, checkpoint):
    """Load TraversalPDE weights from final, periodic, or combined checkpoints."""
    def looks_like_traversal_state(state_dict):
        return any(
            key == 'c'
            or key.startswith('F.')
            or key.startswith('PSI')
            or key == 'PSI_SET'
            for key in state_dict.keys()
        )

    state_dict = None
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get('traversal_sets'), dict):
            state_dict = checkpoint['traversal_sets']
        elif isinstance(checkpoint.get('support_sets'), dict):
            state_dict = checkpoint['support_sets']
        elif isinstance(checkpoint.get('state_dict'), dict) and looks_like_traversal_state(checkpoint['state_dict']):
            state_dict = checkpoint['state_dict']
        elif all(isinstance(key, str) for key in checkpoint) and looks_like_traversal_state(checkpoint):
            state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError(
            "Could not find TraversalPDE weights in checkpoint. "
            "Expected 'traversal_sets', 'support_sets', 'state_dict', or a traversal state dict."
        )
    traversal_pde.load_state_dict(state_dict, strict=True)


def unroll_paths(traversal_pde, initial_latent, step_scale, two_sided=False):
    """Unroll all K PDE traversals in parallel and return [batch, K, frames, latent_dim].

    Two-sided traversals are ordered from the farthest negative frame through the
    shared initial latent to the farthest positive frame.
    """
    num_paths = traversal_pde.num_traversal_sets
    half_range = traversal_pde.num_traversal_timesteps // 2
    initial_latents = initial_latent.unsqueeze(1).expand(-1, num_paths, -1).contiguous()
    z_cur = initial_latents.clone()
    positive_latents = [z_cur]
    dt_value = step_scale * 2 / max(1, half_range - 1)
    dt_batch = torch.full(
        (initial_latent.shape[0], 1),
        float(dt_value),
        device=initial_latent.device,
        dtype=initial_latent.dtype,
    )

    with torch.no_grad():
        for step in range(half_range - 1):
            t_batch = torch.full(
                (initial_latent.shape[0], 1),
                float(step),
                device=z_cur.device,
                dtype=z_cur.dtype,
            )
            z_cur, delta_z = traversal_pde.inference(z_cur, t_batch, dt=dt_batch)
            z_cur = z_cur + delta_z
            positive_latents.append(z_cur)

        if two_sided:
            z_cur = initial_latents.clone()
            negative_latents = []
            for step in range(half_range - 1):
                t_batch = torch.full(
                    (initial_latent.shape[0], 1),
                    float(step),
                    device=z_cur.device,
                    dtype=z_cur.dtype,
                )
                z_cur, delta_z = traversal_pde.inference(z_cur, t_batch, dt=dt_batch)
                z_cur = z_cur - delta_z
                negative_latents.append(z_cur)

            path_latents = list(reversed(negative_latents)) + positive_latents
        else:
            path_latents = positive_latents

    return torch.stack(path_latents, dim=2)


def get_concat_h(img_file_orig,
                 shifted_img_file,
                 size,
                 img_id,
                 s,
                 shift_steps,
                 path_id,
                 draw_header=False,
                 draw_progress_bar=True):
    img_orig = Image.open(img_file_orig).resize((size, size))
    img_orig_w = img_orig.width
    img_orig_h = img_orig.height

    img_shifted = Image.open(shifted_img_file).resize((size, size))
    img_shifted_w = img_shifted.width

    dst = Image.new('RGB', (img_orig_w + img_shifted_w, img_orig_h))
    dst.paste(img_orig, (0, 0))
    dst.paste(img_shifted, (img_orig_w, 0))

    # Add header with img_id and path_id
    if draw_header:
        draw = ImageDraw.Draw(dst)
        offset_w = 6
        offset_h = 6
        t_w = 270
        t_h = 13
        draw.rectangle(xy=[(offset_w, offset_h), (offset_w + t_w, offset_h + t_h)], fill=(0, 0, 0))
        draw.text((offset_w + 2, offset_h + 2), "{}/{:03d}".format(img_id, path_id), fill=(255, 255, 255))

    # Draw progress bar
    if draw_progress_bar:
        draw = ImageDraw.Draw(dst)
        bar_h = 7
        bar_color = (252, 186, 3)
        draw.rectangle(xy=[(size, size - bar_h), ((1 + s / shift_steps) * size, size)], fill=bar_color)

    return dst


def main():
    """WarpedGANSpace -- Latent space traversal script.

    A script for traversing the latent space of a pre-trained GAN generator through paths defined by the warpings of
    a set of pre-trained support vectors. Latent codes are sampled from the experiment's training configuration. The
    generated images are stored under the `results/` directory.

    Options:
        ================================================================================================================
        -v, --verbose : set verbose mode on
        ================================================================================================================
        --exp         : set experiment's model dir, as created by `train.py`, i.e., it should contain a sub-directory
                        `models/` with two files, namely `recognizer.pt` and `traversal_sets.pt`, which
                        contain the weights for the recognizer and the traversal sets, respectively, and an `args.json`
                        file that contains the arguments the model has been trained with.
        --shift-leap  : scale the PDE rollout step size.
        --two-sided   : traverse both negative and positive directions from the same initial latent.
        --batch-size  : set generator batch size (if not set, use the total number of images per path)
        --img-size    : set size of saved generated images (if not set, use the output size of the respective GAN
                        generator)
        --img-quality : JPEG image quality (max 95)
        --gif         : generate collated GIF images for all paths and all latent codes
        --gif-size    : set GIF image size
        --gif-fps     : set number of frames per second for the generated GIF images
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="WarpedGANSpace latent space traversal script")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")
    parser.add_argument('--shift-leap', type=float, default=1.0,
                        help="scale the PDE rollout step size")
    parser.add_argument('--two-sided', action='store_true',
                        help="traverse both negative and positive directions from the same initial latent")
    parser.add_argument('--batch-size', type=int, help="set generator batch size (if not set, use the total number of "
                                                       "images per path)")
    parser.add_argument('--img-size', type=int, help="set size of saved generated images (if not set, use the output "
                                                     "size of the respective GAN generator)")
    parser.add_argument('--img-quality', type=int, default=75, help="set JPEG image quality")
    parser.add_argument('--gif', action='store_true', help="Create GIF traversals")
    parser.add_argument('--gif-size', type=int, default=256, help="set gif resolution")
    parser.add_argument('--gif-fps', type=int, default=30, help="set gif frame rate")
    # ================================================================================================================ #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check structure of `args.exp`
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Invalid given directory: {}".format(args.exp))

    # -- models directory (traversal checkpoints)
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError("Invalid models directory: {}".format(models_dir))

    # Device selection (CUDA > MPS > CPU), matching training and pair generation.
    device = choose_device()
    multi_gpu = device.type == 'cuda' and torch.cuda.device_count() > 1

    # Load args + checkpoint metadata.
    a, ckpt, ckpt_path = load_traversal_sets(models_dir, device)
    gan_type = a.__dict__["gan_type"]

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN type: {}".format(gan_type))
        stylegan2_weight_key = (
            'early_output'
            if a.__dict__.get('early_output', False)
            else a.__dict__.get('stylegan2_resolution', 1024)
        )
        print("  \\__Pre-trained weights: {}".format(
            GAN_WEIGHTS[gan_type]['weights'][stylegan2_weight_key]
            if gan_type == 'StyleGAN2' else GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]]))

    G = build_gan(a).to(device).eval()
    if multi_gpu:
        G = DataParallelPassthrough(G)

    # Build traversal model S
    if args.verbose:
        print("#. Build traversal model S...")

    from lib.TraversalPDE import traversal_options
    S = TraversalPDE(num_traversal_sets=a.__dict__["num_traversal_sets"],
                     num_traversal_timesteps=a.__dict__["num_traversal_timesteps"],
                     traversal_vectors_dim=G.dim_z,
                     **traversal_options(a)).to(device).eval()
    if args.verbose:
        print("  \\__Pre-trained weights: {}".format(ckpt_path))
    load_traversal_weights(S, ckpt)
    if args.verbose:
        print("  \\__Set to evaluation mode")

    # Set number of generative paths
    num_gen_paths = S.num_traversal_sets

    # Create output dir for generated images
    out_dir = osp.join(args.exp, 'results')
    os.makedirs(out_dir, exist_ok=True)

    # Set default batch size
    if args.batch_size is None:
        args.batch_size = num_gen_paths
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    # Sample latents using the same Z/W-space and truncation handling as training.
    zs = sample_z(50, G, params=a, device=device)
    num_of_latent_codes = zs.size()[0]

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                            [Latent space traversal]                                            ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    if args.verbose:
        one_sided_frames = max(1, S.num_traversal_timesteps // 2)
        num_path_frames = 2 * one_sided_frames - 1 if args.two_sided else one_sided_frames
        print("#. Traverse latent space...")
        print("  \\__Experiment       : {}".format(osp.basename(osp.abspath(args.exp))))
        print("  \\__Traversal mode   : {}".format("two-sided" if args.two_sided else "one-sided"))
        print("  \\__PDE path frames  : {}".format(num_path_frames))
        print("  \\__PDE step scale   : {}".format(args.shift_leap))
        print("  \\__Save results at  : {}".format(out_dir))

    # Iterate over given latent codes
    for i in range(num_of_latent_codes):
        #if i<=11:
        #    continue
        # Un-squeeze current latent code in shape [1, dim] and create its output ID.
        z_ = zs[i, :].unsqueeze(0)

        latent_code_hash = 'sample_{:06d}'.format(i)
        if args.verbose:
            update_progress("  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash,
                                                                                  i+1,
                                                                                  num_of_latent_codes),
                            num_of_latent_codes, i)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Create directory for storing path images
        transformed_images_root_dir = osp.join(latent_code_dir, 'paths_images')
        os.makedirs(transformed_images_root_dir, exist_ok=True)

        ## ========================================================================================================== ##
        ##                                                                                                            ##
        ##                                             [ Path Traversal ]                                             ##
        ##                                                                                                            ##
        ## ========================================================================================================== ##
        # TraversalPDE expands the base latent and advances all K paths in parallel.
        paths_latent_codes = unroll_paths(S, z_, args.shift_leap, two_sided=args.two_sided).squeeze(0)
        num_path_frames = paths_latent_codes.shape[1]
        original_frame_index = num_path_frames // 2 if args.two_sided else 0

        # Preserve the existing per-path directory layout used by downstream scripts.
        transformed_images_dirs = []
        for dim in range(num_gen_paths):
            path_dir = osp.join(transformed_images_root_dir, 'path_{:03d}'.format(dim))
            os.makedirs(path_dir, exist_ok=True)
            transformed_images_dirs.append(path_dir)

        # Generate in bounded batches and save immediately so high-resolution traversals do not retain every image.
        flat_path_latents = paths_latent_codes.reshape(-1, G.dim_z)
        for batch_start in range(0, flat_path_latents.shape[0], args.batch_size):
            latent_batch = flat_path_latents[batch_start:batch_start + args.batch_size]
            with torch.no_grad():
                transformed_batch = G(latent_batch).cpu()

            for batch_index, image_tensor in enumerate(transformed_batch):
                flat_index = batch_start + batch_index
                dim, t = divmod(flat_index, num_path_frames)
                transformed_image = tensor2image(
                    image_tensor,
                    img_size=args.img_size,
                    adaptive=True,
                )
                transformed_image.save(osp.join(transformed_images_dirs[dim], '{:06d}.jpg'.format(t)),
                                       "JPEG", quality=args.img_quality, optimize=True, progressive=True)
                if t == original_frame_index and dim == 0:
                    transformed_image.save(osp.join(latent_code_dir, 'original_image.jpg'),
                                           "JPEG", quality=95, optimize=True, progressive=True)
                if args.verbose and t == num_path_frames - 1:
                    update_progress(
                        "      \\__path: {:03d}/{:03d} ".format(dim + 1, num_gen_paths),
                        num_gen_paths,
                        dim + 1,
                    )
        # ============================================================================================================ #

        # Save all latent paths for the current sample in [K, frames, latent_dim].
        torch.save(paths_latent_codes.detach().cpu(), osp.join(latent_code_dir, 'paths_latent_codes.pt'))

        if args.verbose:
            update_stdout(1)
            print()
            print()

    # Collate traversal GIFs
    if args.gif:
        # Build results file structure
        structure = dict()
        generated_img_subdirs = [dI for dI in os.listdir(out_dir) if os.path.isdir(osp.join(out_dir, dI)) and
                                 dI not in ('paths_gifs', 'validation_results')]
        generated_img_subdirs.sort()
        for img_id in generated_img_subdirs:
            structure.update({img_id: {}})
            path_images_dir = osp.join(out_dir, '{}'.format(img_id), 'paths_images')
            path_images_subdirs = [dI for dI in os.listdir(path_images_dir)
                                   if os.path.isdir(os.path.join(path_images_dir, dI))]
            path_images_subdirs.sort()
            for item in path_images_subdirs:
                structure[img_id].update({item: [dI for dI in os.listdir(osp.join(path_images_dir, item))
                                                 if osp.isfile(os.path.join(path_images_dir, item, dI))]})

        # Create directory for storing traversal GIFs
        os.makedirs(osp.join(out_dir, 'paths_gifs'), exist_ok=True)

        # For each interpretable path (warping function), collect the generated image sequences for each original latent
        # code and collate them into a GIF file
        print("#. Collate GIFs...")
        num_of_frames = list()
        for dim in range(num_gen_paths):
            if args.verbose:
                update_progress("  \\__path: {:03d}/{:03d} ".format(dim + 1, num_gen_paths), num_gen_paths, dim + 1)

            gif_frames = []
            for img_id in structure.keys():
                original_img_file = osp.join(out_dir, '{}'.format(img_id), 'original_image.jpg')
                shifted_images_dir = osp.join(out_dir, '{}'.format(img_id), 'paths_images', 'path_{:03d}'.format(dim))

                row_frames = []
                img_id_num_of_frames = 0
                for t in range(len(structure[img_id]['path_{:03d}'.format(dim)])):
                    img_id_num_of_frames += 1
                for t in range(len(structure[img_id]['path_{:03d}'.format(dim)])):
                    shifted_img_file = osp.join(shifted_images_dir, '{:06d}.jpg'.format(t))

                    # Concatenate `original_img_file` and `shifted_img_file`
                    row_frames.append(get_concat_h(img_file_orig=original_img_file,
                                                   shifted_img_file=shifted_img_file,
                                                   size=args.gif_size,
                                                   img_id=img_id,
                                                   s=t,
                                                   shift_steps=img_id_num_of_frames,
                                                   path_id=dim))
                num_of_frames.append(img_id_num_of_frames)
                gif_frames.append(row_frames)

            if len(set(num_of_frames)) > 1:
                print("#. Warning: Inconsistent number of frames for image sequences: {}".format(num_of_frames))

            # Create full GIF frames
            full_gif_frames = []
            for f in range(int(num_of_frames[0])):
                gif_f = Image.new('RGB', (2 * args.gif_size, len(structure) * args.gif_size))
                for i in range(len(structure)):
                    gif_f.paste(gif_frames[i][f], (0, i * args.gif_size))
                full_gif_frames.append(gif_f)

            # Save gif
            im = Image.new(mode='RGB', size=(2 * args.gif_size, len(structure) * args.gif_size))
            im.save(
                fp=osp.join(out_dir, 'paths_gifs', 'path_{:03d}.gif'.format(dim)),
                append_images=full_gif_frames,
                save_all=True,
                optimize=True,
                loop=0,
                duration=1000 // args.gif_fps)


if __name__ == '__main__':
    main()

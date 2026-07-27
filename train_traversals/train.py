import argparse
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2,build_stylegan2mps, build_sngan
from torch import nn
from lib.aux import choose_device

def main():
    """PotentialFlow -- Training script.

    Options:
        ===[ Pre-trained GAN Generator (G) ]============================================================================
        --gan-type                 : set pre-trained GAN type
        --z-truncation             : set latent code sampling truncation parameter. If set, latent codes will be sampled
                                     from a standard Gaussian distribution truncated to the range [-args.z_truncation,
                                     +args.z_truncation]
        --biggan-target-classes    : set list of classes to use for conditional BigGAN (see BIGGAN_CLASSES in
                                     lib/config.py). E.g., --biggan-target-classes 14 239.
        --stylegan2-resolution     : set StyleGAN2 generator output images resolution:  256 or 1024 (default: 1024)
        --shift-in-w-space         : search latent paths in StyleGAN2's W-space (otherwise, look in Z-space)

        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-traversal-sets     : set number of support sets; i.e., number of warping functions -- number of
                                     interpretable paths
        -D, --num-traversal-timesteps  : set number of support dipoles per support set

        --traversal-set-lr           : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --reconstructor-lr         : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --lambda-cls               : classification loss weight
        --lambda-reg               : regression loss weight
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard

    """
    parser = argparse.ArgumentParser(description="Potential flow training script for pre-trained GANs")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan-type', type=str, choices=GAN_WEIGHTS.keys(), help='set GAN generator model type')
    parser.add_argument('--z-truncation', type=float, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),
                        help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-traversal-sets', type=int, help="set number of support sets (potential functions)")
    parser.add_argument('-D', '--num-traversal-timesteps', type=int, help="set number of timesteps per potential")
    parser.add_argument('--traversal-set-lr', type=float, default=3e-4, help="set learning rate")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-lr', type=float, default=3e-4,
                        help="set learning rate for reconstructor R optimization")
    parser.add_argument('--reconstructor-type', type=str, default='ResNet',
                        help='set reconstructor network type')

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="set batch size")
    parser.add_argument('--accumulate-grad-steps', type=int, default=1, help="set number of steps to accumulate gradients")
    parser.add_argument('--warmup-fraction', type=float, default=0.05, help="warmup fraction")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=.0, help="regression loss weight")
    parser.add_argument('--lambda-pde', type=float, default=1.0, help="pde loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    # === Validation ===================================================================================================== #
    parser.add_argument('--val-freq', type=int, default=10, help="set number iterations per validation")
    # === Restart ===================================================================================================== #
    parser.add_argument('--new-experiment', action='store_true',default=False, help='set to True to start a new experiment')
    parser.add_argument('--reset_lr', action='store_true', help="reset learning rate")
    parser.add_argument('--reset_weight_decay', action='store_true', help="reset weight decay")
    parser.add_argument('--reset_schedulers', action='store_true', help="reset schedulers")
    parser.add_argument('--reset_start_iter', action='store_true', help="reset start iteration")


    # Parse given arguments
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args, new_experiment=args.new_experiment)

    # Device selection (CUDA > MPS > CPU)
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    multi_gpu = use_cuda and (torch.cuda.device_count() > 1)
    reconstructor_pool_size = 2 if args.reconstructor_type == 'LeNet' and args.gan_type != 'SNGAN_AnimeFaces' else 1

    device = choose_device()

    # Set default tensor type for CUDA only (no MPS default tensor type exists)
    torch.set_default_device(device)

    # Build GAN generator model and load with pre-trained weights
    print("#. Build GAN generator model G and load with pre-trained weights...")
    print("  \\__GAN type: {}".format(args.gan_type))
    if args.gan_type == 'StyleGAN2':
        print("  \\__Search for paths in {}-space".format('W' if args.shift_in_w_space else 'Z'))
    if args.z_truncation:
        print("  \\__Input noise truncation: {}".format(args.z_truncation))
    print("  \\__Pre-trained weights: {}".format(
        GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution] if args.gan_type == 'StyleGAN2' else
        GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]]))

    # === BigGAN ===
    if args.gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                         target_classes=args.biggan_target_classes)
        # print(G.device,G)
        # print(G(torch.randn(1, 512).to(G.device)))
    # === ProgGAN ===
    elif args.gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]])
    # === StyleGAN ===
    elif args.gan_type == 'StyleGAN2':
        # TODO: remove this once the StyleGAN2Wrapper is fixed
        if use_mps:
            G = build_stylegan2mps(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution],
                            resolution=args.stylegan2_resolution,
                            shift_in_w_space=args.shift_in_w_space)
        else:   
            G = build_stylegan2(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution],
                            resolution=args.stylegan2_resolution,
                            shift_in_w_space=args.shift_in_w_space)
        if args.stylegan2_resolution == 1024:
            reconstructor_pool_size = 4
    # === Spectrally Normalised GAN (SNGAN) ===
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                        gan_type=args.gan_type)

    # Build Potentials model (legacy: Support Sets) S
    print("#. Build Potentials (Support Sets) S...")
    print("  \\__Number of Potentials    : {}".format(args.num_traversal_sets))
    print("  \\__Number of Timesteps : {}".format(args.num_traversal_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.dim_z))

    S = TraversalPDE(num_traversal_sets=args.num_traversal_sets,
                    num_traversal_timesteps=args.num_traversal_timesteps,
                    traversal_vectors_dim=G.dim_z,
                    lambdas={'BB': 0.25, 'signed_g2orth': 1.0},
                    ) 

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build recognizer model (legacy: reconstructor) R
    print("#. Build recognizer (reconstructor) model R...")
    R = Recognizer(reconstructor_type=args.reconstructor_type,
                      dim_index=S.num_traversal_sets,
                      dim_time=S.num_traversal_timesteps,
                      channels=1 if args.gan_type == 'SNGAN_MNIST' else 3,
                      pool_size=reconstructor_pool_size)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    trn = TrainerPotential(params=args, exp_dir=exp_dir, device=device, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, traversal_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()

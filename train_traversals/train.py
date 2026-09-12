import argparse
from lib.TraversalPDE import add_traversal_arguments, traversal_options
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2mps, build_sngan
from torch import nn
from lib.utils import choose_device
from lib.val_utils import add_validation_arguments
from lib.recognizer import add_recognizer_arguments, recognizer_options

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

        ===[ recognizer (R) ]========================================================================================
        --recognizer-type       : set recognizer network type
        --recognizer-lr         : set learning rate for recognizer R optimization

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
    parser.add_argument('--z-truncation', type=float, default=1.0, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),
                        help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-traversal-sets', type=int, help="set number of support sets (potential functions)")
    parser.add_argument('-D', '--num-traversal-timesteps', type=int, help="set number of timesteps per potential")
    parser.add_argument('--traversal-set-lr', type=float, default=3e-4, help="set learning rate")

    # === recognizer (R) ========================================================================================== #
    parser.add_argument('--recognizer-lr', type=float, default=3e-4,
                        help="set learning rate for recognizer R optimization")
    parser.add_argument('--recognizer-type', type=str, default='ResNet',
                        help='set recognizer network type')
    add_recognizer_arguments(parser)

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="set batch size")
    parser.add_argument('--accumulate-grad-steps', type=int, default=1, help="set number of steps to accumulate gradients")
    parser.add_argument('--generator-recompute-chunk', type=int, default=0,
                        help="bound frozen deterministic generator activations by replaying chunks in backward; 0 disables")
    parser.add_argument('--warmup-fraction', type=float, default=0.05, help="warmup fraction")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=.0, help="regression loss weight")
    parser.add_argument('--lambda-pde', type=float, default=1.0, help="pde loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "bf16"],
        help="generator and recognizer mixed precision (no or bf16)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the GAN generator (synthesis only for StyleGAN)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        help="torch.compile mode (default, reduce-overhead, max-autotune)",
    )
    parser.add_argument('--track-dt-stats', action='store_true',
                        help="cache per-step dt statistics (adds an accelerator synchronization)")
    # === Validation ===================================================================================================== #
    add_validation_arguments(parser)
    # === Restart ===================================================================================================== #
    parser.add_argument('--new-experiment', action='store_true',default=False, help='set to True to start a new experiment')
    parser.add_argument('--reset_lr', action='store_true', help="reset learning rate")
    parser.add_argument('--reset_weight_decay', action='store_true', help="reset weight decay")
    parser.add_argument('--reset_schedulers', action='store_true', help="reset schedulers")
    parser.add_argument('--reset_start_iter', action='store_true', help="reset start iteration")


    # Parse given arguments
    add_traversal_arguments(parser)
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args, new_experiment=args.new_experiment)

    # Device selection (CUDA > MPS > CPU)
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    multi_gpu = use_cuda and (torch.cuda.device_count() > 1)
    recognizer_pool_size = 2 if args.recognizer_type.endswith('LeNet') and args.gan_type != 'SNGAN_AnimeFaces' else 1

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
    generator_runtime_options = dict(
        mixed_precision=args.mixed_precision,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )
    if args.gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                         target_classes=args.biggan_target_classes,
                         **generator_runtime_options)
        # print(G.device,G)
        # print(G(torch.randn(1, 512).to(G.device)))
    # === ProgGAN ===
    elif args.gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                          **generator_runtime_options)
    # === StyleGAN ===
    elif args.gan_type == 'StyleGAN2':
        G = build_stylegan2mps(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution],
                            resolution=args.stylegan2_resolution,
                            shift_in_w_space=args.shift_in_w_space,
                            use_optimized=args.compile,
                            **generator_runtime_options)
        if args.stylegan2_resolution == 1024:
            recognizer_pool_size = 4
    # === Spectrally Normalised GAN (SNGAN) ===
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                        gan_type=args.gan_type,
                        **generator_runtime_options)

    # Build Potentials model (legacy: Support Sets) S
    print("#. Build Potentials (Support Sets) S...")
    print("  \\__Number of Potentials    : {}".format(args.num_traversal_sets))
    print("  \\__Number of Timesteps : {}".format(args.num_traversal_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.dim_z))

    S = TraversalPDE(num_traversal_sets=args.num_traversal_sets,
                    num_traversal_timesteps=args.num_traversal_timesteps,
                    traversal_vectors_dim=G.dim_z,
                    lambdas={'BB': 0.25, 'signed_g2orth': 1.0},
                    **traversal_options(args),
                    ) 

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build recognizer model (legacy: recognizer) R
    print("#. Build recognizer (recognizer) model R...")
    recognizer_cls = AntisymmetricRecognizer if args.recognizer_type.startswith('Antisymmetric') else Recognizer
    recognizer_backbone = args.recognizer_type.removeprefix('Antisymmetric')
    R = recognizer_cls(recognizer_type=recognizer_backbone,
                       dim_index=S.num_traversal_sets,
                       channels=1 if args.gan_type == 'SNGAN_MNIST' else 3,
                       pool_size=args.recognizer_pool_size if args.recognizer_pool_size is not None else recognizer_pool_size,
                       **recognizer_options(args, args.stylegan2_resolution if args.gan_type == 'StyleGAN2'
                                            else GAN_RESOLUTIONS[args.gan_type]))

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    trn = TraversalTrainer(params=args, exp_dir=exp_dir, device=device, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, traversal_sets=S, recognizer=R)


if __name__ == '__main__':
    main()

import argparse
from pathlib import Path

import torch
from lib import (
    GAN_WEIGHTS,
    Reconstructor,
    TrainerPotential,
    WavePDE,
    create_exp_dir,
    load_sd_vae_generator,
)

def main():
    """PotentialFlow -- Training script.

    Options:
        ===[ Pre-trained GAN Generator (G) ]============================================================================
        --gan-type                 : set pre-trained GAN type
        --z-truncation             : set latent code sampling truncation parameter. If set, latent codes will be sampled
                                     from a standard Gaussian distribution truncated to the range [-args.z_truncation,
                                     +args.z_truncation]

        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-support-sets     : set number of support sets; i.e., number of warping functions -- number of
                                     interpretable paths
        -D, --num-support-timesteps  : set number of support dipoles per support set

        --support-set-lr           : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --min-shift-magnitude      : set minimum shift magnitude
        --max-shift-magnitude      : set maximum shift magnitude
        --reconstructor-lr         : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --lambda-cls               : classification loss weight
        --lambda-reg               : regression loss weight
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard

        ===[ Device ]===================================================================================================
        --cuda                     : use CUDA during training (default)
        --no-cuda                  : do NOT use CUDA during training
        --mps                      : use Apple Metal (MPS) backend
        --no-mps                   : do NOT use MPS backend
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="Potential flow training script for pre-trained GANs")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan-type', type=str, default='SD-VAE',
                        choices=list(GAN_WEIGHTS.keys()) + ['SD-VAE'],
                        help='set generator model type')
    parser.add_argument('--z-truncation', type=float, default=1.0,
                        help="set latent code sampling truncation parameter")
    parser.add_argument('--vae-config', type=str, default='../train_eqvae/configs/eqvae_config.yaml',
                        help='path to SD-VAE config')
    parser.add_argument('--vae-ckpt', type=str, default='../train_eqvae/pretrained_models/model.ckpt',
                        help='path to downloaded SD-VAE checkpoint')
    parser.add_argument('--vae-scaling-factor', type=float, default=1.0,
                        help='latent scaling factor to invert before VAE decode')

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of support sets (potential functions)")
    parser.add_argument('-D', '--num-support-timesteps', type=int, help="set number of timesteps per potential")
    parser.add_argument('--support-set-lr', type=float, default=3e-4, help="set learning rate")
    parser.add_argument('--only-potential', type=bool, default=True, help="only train potential")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-lr', type=float, default=2e-4,
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
    parser.add_argument('--lambda-pde', type=float, default=1.00, help="pde loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
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
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    use_cuda = cuda_available
    use_mps = mps_available
    device = torch.device('cuda' if use_cuda else ('mps' if use_mps else 'cpu'))

    # Set default tensor type for CUDA only (no MPS default tensor type exists)
    if use_cuda:
        torch.set_default_device(torch.device('cuda'))
    elif use_mps:
        torch.set_default_device(torch.device('mps'))
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_device(torch.device('cpu'))

    multi_gpu = use_cuda and (torch.cuda.device_count() > 1)

    script_dir = Path(__file__).resolve().parent
    vae_config = (script_dir / args.vae_config).resolve()
    vae_ckpt = (script_dir / args.vae_ckpt).resolve()
    print("#. Load SD-VAE generator...")
    print(f"  \\__Config     : {vae_config}")
    print(f"  \\__Checkpoint : {vae_ckpt}")
    G = load_sd_vae_generator(
        config_path=vae_config,
        ckpt_path=vae_ckpt,
        scaling_factor=args.vae_scaling_factor,
    )

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Potentials    : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps : {}".format(args.num_support_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.dim_z))

    S = WavePDE(num_support_sets=args.num_support_sets,
                    num_support_timesteps=args.num_support_timesteps,
                    support_vectors_dim=G.dim_z,
                    only_potential = args.only_potential,
                    lambdas={'BB':.5, 'g2orth': 1.0},
                    ) 

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    R = Reconstructor(reconstructor_type=args.reconstructor_type,
                      dim_index=S.num_support_sets,
                      dim_time=S.num_support_timesteps,
                      channels=3,
                      pool_size=1)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    print("  \\__Only train potential: {}".format(args.only_potential))
    trn = TrainerPotential(params=args, exp_dir=exp_dir, device=device, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()

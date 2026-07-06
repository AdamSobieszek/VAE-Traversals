import argparse
from pathlib import Path

import torch
from lib import (
    GAN_RESOLUTIONS,
    GAN_WEIGHTS,
    Recognizer,
    TrainerPotential,
    TraversalPDE,
    create_exp_dir,
)
from models.gan_load import build_gat


def resolve_gat_checkpoint(args, script_dir):
    gan_cfg = GAN_WEIGHTS[args.gan_type]
    resolution = args.resolution or GAN_RESOLUTIONS[args.gan_type]
    if args.gat_ckpt:
        ckpt_path = Path(args.gat_ckpt).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (script_dir / ckpt_path).resolve()
    else:
        ckpt_path = (script_dir / gan_cfg["weights"][resolution]).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"GAT checkpoint not found: {ckpt_path}. "
            "Run scripts/setup_gat_inference.sh or pass --gat-ckpt."
        )
    return ckpt_path, resolution, gan_cfg


def load_gat_generator(args, script_dir, device):
    ckpt_path, resolution, gan_cfg = resolve_gat_checkpoint(args, script_dir)
    print("#. Load GAT generator...")
    print(f"  \\__Checkpoint : {ckpt_path}")
    print(f"  \\__Model      : {args.gat_model or gan_cfg.get('model', 'GAT-XL/2')}")
    print(f"  \\__Resolution : {resolution}")
    print(f"  \\__Precision  : {args.mixed_precision}")

    return build_gat(
        str(ckpt_path),
        target_classes=tuple(args.target_classes),
        model_name=args.gat_model or gan_cfg.get("model"),
        resolution=resolution,
        vae_variant=args.vae_variant or gan_cfg.get("vae_variant", "ema"),
        truncation_psi=args.truncation_psi,
        mixed_precision=args.mixed_precision,
        load_vae=True,
    ).to(device)


def main():
    """Potential-flow training with a frozen GAT generator in SD-VAE latent space."""
    parser = argparse.ArgumentParser(description="Potential flow training script for GAT")

    # === Pre-trained GAT Generator (G) ============================================================ #
    parser.add_argument(
        "--gan-type",
        type=str,
        default="GAT",
        choices=[key for key in GAN_WEIGHTS.keys() if key == "GAT"],
        help="pretrained generator type (GAT only in this script)",
    )
    parser.add_argument(
        "--gat-ckpt",
        type=str,
        default="",
        help="path to GAT checkpoint (defaults to GAN_WEIGHTS[GAT] entry)",
    )
    parser.add_argument(
        "--gat-model",
        type=str,
        default="",
        help="override GAT architecture tag, e.g. GAT-XL/2",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=GAN_RESOLUTIONS["GAT"],
        help="output image resolution",
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="+",
        default=[239],
        help="ImageNet class ids sampled during GAT decoding",
    )
    parser.add_argument(
        "--vae-variant",
        type=str,
        default=GAN_WEIGHTS["GAT"].get("vae_variant", "ema"),
        choices=["ema", "mse"],
        help="Stable Diffusion VAE variant used to decode GAT latents",
    )
    parser.add_argument(
        "--truncation-psi",
        type=float,
        default=0.8,
        help="GAT style-vector truncation strength",
    )
    parser.add_argument(
        "--z-truncation",
        type=float,
        default=1.0,
        help="latent code sampling truncation before warping",
    )

    # === Support Sets (S) ======================================================================= #
    parser.add_argument("-K", "--num-support-sets", type=int, default=8,
                        help="number of support sets (potential functions)")
    parser.add_argument("-D", "--num-support-timesteps", type=int, default=4,
                        help="number of timesteps per potential")
    parser.add_argument("--support-set-lr", type=float, default=3e-4, help="support-set learning rate")
    parser.add_argument("--only-potential", type=bool, default=True, help="only train potential")

    # === Reconstructor (R) ====================================================================== #
    parser.add_argument("--recognizer-lr", type=float, default=3e-4,
                        help="learning rate for recognizer optimization")
    parser.add_argument("--recognizer-type", type=str, default="ResNet",
                        help="recognizer network type")

    # === Training =============================================================================== #
    parser.add_argument("--max-iter", type=int, default=100000, help="maximum training iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--accumulate-grad-steps", type=int, default=1,
                        help="gradient accumulation steps")
    parser.add_argument("--warmup-fraction", type=float, default=0.05, help="warmup fraction")
    parser.add_argument("--lambda-cls", type=float, default=1.00, help="classification loss weight")
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="regression loss weight")
    parser.add_argument("--lambda-pde", type=float, default=1.00, help="PDE loss weight")
    parser.add_argument("--log-freq", default=10, type=int, help="logging frequency")
    parser.add_argument("--ckp-freq", default=1000, type=int, help="checkpoint frequency")
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "bf16"],
        help="shared precision for frozen GAT and trainable recognizer",
    )

    # === Restart ================================================================================ #
    parser.add_argument("--new-experiment", action="store_true", default=False,
                        help="start a new experiment directory")
    parser.add_argument("--reset_lr", action="store_true", help="reset learning rate")
    parser.add_argument("--reset_weight_decay", action="store_true", help="reset weight decay")
    parser.add_argument("--reset_schedulers", action="store_true", help="reset schedulers")
    parser.add_argument("--reset_start_iter", action="store_true", help="reset start iteration")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GAT traversal training requires CUDA.")

    exp_dir = create_exp_dir(args, new_experiment=args.new_experiment)

    device = torch.device("cuda")
    torch.set_default_device(device)
    multi_gpu = torch.cuda.device_count() > 1

    script_dir = Path(__file__).resolve().parent
    G = load_gat_generator(args, script_dir, device)

    print("#. Build Support Sets S...")
    print("  \\__Number of Potentials : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps  : {}".format(args.num_support_timesteps))
    print("  \\__Support Vectors dim  : {}".format(G.dim_z))

    S = TraversalPDE(
        num_support_sets=args.num_support_sets,
        num_support_timesteps=args.num_support_timesteps,
        support_vectors_dim=G.dim_z,
        only_potential=args.only_potential,
        lambdas={"BB": 0.2, "signed_g2orth": 1.0},
    )

    print("  \\__Trainable parameters: {:,}".format(
        sum(p.numel() for p in S.parameters() if p.requires_grad)
    ))

    print("#. Build recognizer model R...")
    R = Recognizer(
        recognizer_type=args.recognizer_type,
        dim_index=S.num_support_sets,
        channels=4 if args.gan_type == "GAT" else 3,
        pool_size=1,
    )
    print("  \\__Channels: {}".format(4 if args.gan_type == "GAT" else 3))
    print("  \\__Trainable parameters: {:,}".format(
        sum(p.numel() for p in R.parameters() if p.requires_grad)
    ))

    print("#. Experiment: {}".format(exp_dir))
    print("  \\__Only train potential: {}".format(args.only_potential))
    trn = TrainerPotential(params=args, exp_dir=exp_dir, device=device, multi_gpu=multi_gpu)
    trn.train(generator=G, support_sets=S, recognizer=R)


if __name__ == "__main__":
    main()

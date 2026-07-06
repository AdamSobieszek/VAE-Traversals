import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
from pathlib import Path
from collections import OrderedDict
import json

import torch
import torch.utils.checkpoint

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from models.generator import CAT_models
from models.discriminator import CATD_models
from losses import CATLoss
from utils import load_encoders

from dataset import CustomDataset, CustomDataset_DiT
from diffusers.models import AutoencoderKL

import math
from torchvision.utils import make_grid

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

logger = get_logger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def save_sample_sheet(out_samples, save_dir, epoch, global_step, suffix=""):
    from PIL import Image

    sheet_dir = os.path.join(save_dir, "sample_sheets")
    os.makedirs(sheet_dir, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    path = os.path.join(
        sheet_dir,
        f"epoch_{epoch:04d}_step_{global_step:07d}{tag}_sheet.png",
    )
    Image.fromarray(array2grid(out_samples)).save(path)
    return path


@torch.no_grad()
def sample_posterior(moments, latents_scale=1.0, latents_bias=0.0):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * latents_scale + latents_bias


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if name in ema_buffers:
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def build_train_dataset(args):
    if args.resolution == 256:
        return CustomDataset(args.data_dir)

    if args.resolution == 128:
        if args.dit_features_dir is None or args.dit_labels_dir is None:
            raise ValueError(
                "128-resolution training requires --dit-features-dir and --dit-labels-dir."
            )
        return CustomDataset_DiT(
            features_dir=args.dit_features_dir,
            labels_dir=args.dit_labels_dir,
        )

    raise ValueError(f"Unsupported resolution: {args.resolution}")


def parse_cons_weights(cons_weights_str):
    return tuple(float(x.strip()) for x in cons_weights_str.split(","))


def uses_wandb(report_to):
    if report_to is None:
        return False
    trackers = [tracker.strip().lower() for tracker in str(report_to).split(",")]
    return "wandb" in trackers


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    log_with = None if str(args.report_to).lower() in ("none", "no", "false", "") else args.report_to

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[
            DistributedDataParallelKwargs(broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(seconds=5400)),
        ],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, "w") as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

        if args.resume_step == 0:
            os.makedirs(os.path.join(save_dir, "source_codes"), exist_ok=True)
            shutil.copytree(
                os.getcwd(),
                os.path.join(save_dir, "source_codes"),
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "wandb", "_experiments", "features_temp", "temp", "pretrained_models"
                ),
            )

    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    with accelerator.main_process_first():
        if args.enc_type != "None":
            encoders, encoder_types, architectures = load_encoders(
                args.enc_type, device, args.resolution
            )
        else:
            encoders, encoder_types, architectures = [], [], []

    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != "None" else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    if args.model not in CAT_models:
        raise KeyError(f"Unknown generator model: {args.model}")
    if args.modelD not in CATD_models:
        raise KeyError(f"Unknown discriminator model: {args.modelD}")

    generator = CAT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dims=z_dims,
        **block_kwargs,
    )
    discriminator = CATD_models[args.modelD](
        num_classes=args.num_classes,
        z_dims=z_dims,
        **block_kwargs,
    )

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    ema = deepcopy(generator).to(device)

    with accelerator.main_process_first():
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.zeros(1, 4, 1, 1).to(device)

    cons_weights = parse_cons_weights(args.cons_weights)
    loss_fn = CATLoss(
        encoders=encoders,
        encoder_types=encoder_types,
        architectures=architectures,
        accelerator=accelerator,
        r1_every=args.R1_every,
        r1_gamma=args.R1_gamma,
        r2_every=args.R2_every,
        r2_gamma=args.R2_gamma,
        lambda_repa=args.lambda_repa,
        lambda_cons=args.lambda_cons,
        cons_weights=cons_weights,
        gp_eps=args.gp_eps,
        gp_batch_frac=args.gp_batch_frac,
    )

    if accelerator.is_main_process:
        generator_params = sum(p.numel() for p in generator.parameters())
        discriminator_params = sum(p.numel() for p in discriminator.parameters())
        discriminator_aux = getattr(discriminator, "proj", None)
        discriminator_aux_params = (
            sum(p.numel() for p in discriminator_aux.parameters())
            if discriminator_aux is not None
            else 0
        )
        discriminator_backbone_params = discriminator_params - discriminator_aux_params
        logger.info(f"CAT generator parameters: {generator_params:,}")
        logger.info(
            f"CAT discriminator backbone parameters: {discriminator_backbone_params:,}"
        )
        if discriminator_aux_params > 0:
            logger.info(
                f"CAT discriminator auxiliary parameters: {discriminator_aux_params:,}"
            )
        logger.info(f"CAT discriminator total parameters: {discriminator_params:,}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    glr = args.learning_rate
    dlr = args.learning_rate

    optimizerG = torch.optim.AdamW(
        generator.parameters(),
        lr=glr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizerD = torch.optim.AdamW(
        discriminator.parameters(),
        lr=dlr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = build_train_dataset(args)

    if args.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f"--batch-size ({args.batch_size}) must be divisible by num_processes "
            f"({accelerator.num_processes})."
        )
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    if local_batch_size < 1:
        raise ValueError(
            f"Local batch size must be at least 1; got {local_batch_size} from "
            f"batch_size={args.batch_size}, num_processes={accelerator.num_processes}."
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    global_step = 0
    fid_best = 1e5
    fid_cur = 1e5

    if args.resume_step != 0:
        ckpt_name = "latest.pt" if args.resume_step < 0 else str(args.resume_step).zfill(7) + ".pt"
        ckpt = torch.load(
            f"{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}",
            map_location="cpu",
            weights_only=False,
        )
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        ema.load_state_dict(ckpt["ema"])
        optimizerG.load_state_dict(ckpt["optG"])
        optimizerD.load_state_dict(ckpt["optD"])
        global_step = ckpt["steps"]
        wandb_run_id = None
        is_resume = "allow"
    else:
        wandb_run_id = None
        is_resume = None

    generator, discriminator, optimizerG, optimizerD, train_dataloader = accelerator.prepare(
        generator, discriminator, optimizerG, optimizerD, train_dataloader
    )

    update_ema(ema, accelerator.unwrap_model(generator), decay=0)

    generator.train()
    discriminator.train()
    ema.eval()

    use_wandb = uses_wandb(args.report_to)
    if use_wandb and wandb is None:
        raise ImportError("wandb is required when --report-to includes 'wandb'.")
    if accelerator.is_main_process and log_with is not None:
        tracker_config = vars(copy.deepcopy(args))
        init_kwargs = {}
        if use_wandb:
            init_kwargs["wandb"] = {
                "dir": save_dir,
                "name": f"{args.exp_name}",
                "id": wandb_run_id,
                "resume": is_resume,
            }
        accelerator.init_trackers(
            project_name=args.wandb_name,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    sample_batch_size = max(1, 64 // accelerator.num_processes)
    ys_vis = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device).repeat(
        max(1, sample_batch_size // 8)
    )
    n = ys_vis.size(0)
    zs_vis = torch.randn(size=(n, accelerator.unwrap_model(generator).latent_size), device=device)
    xs_vis = torch.randn((n, 4, latent_size, latent_size), device=device)
    stats_metrics = dict()

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)

            with torch.no_grad():
                if x.shape[1] == 8:
                    x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            model_kwargs = dict(y=y)
            with accelerator.accumulate(discriminator):
                accelerator.unwrap_model(discriminator).requires_grad_(True)
                loss, disc_loss_dict, _ = loss_fn.step_disc(
                    generator, discriminator, None, x, raw_image, global_step, model_kwargs
                )
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                optimizerD.step()
                optimizerD.zero_grad(set_to_none=True)
                accelerator.unwrap_model(discriminator).requires_grad_(False)

            with accelerator.accumulate(generator):
                accelerator.unwrap_model(generator).requires_grad_(True)
                next_global_step = global_step + 1 if accelerator.sync_gradients else global_step
                log_alignment = (
                    args.alignment_logging_steps > 0
                    and accelerator.sync_gradients
                    and next_global_step % args.alignment_logging_steps == 0
                )
                loss, gen_loss_dict, _ = loss_fn.step_gen(
                    generator,
                    discriminator,
                    None,
                    x,
                    raw_image,
                    global_step,
                    model_kwargs,
                    log_alignment=log_alignment,
                )
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
                optimizerG.step()
                optimizerG.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, accelerator.unwrap_model(generator), decay=args.ema_decay)
                accelerator.unwrap_model(generator).requires_grad_(False)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                from metrics import metric_main

                if accelerator.is_main_process:
                    print("Evaluating metrics...")
                for metric in ["fid5k"]:
                    with torch.no_grad():
                        result_dict = metric_main.calc_metric(
                            metric=metric,
                            G=ema,
                            vae=vae,
                            latent_bias=latents_bias,
                            latent_scale=latents_scale,
                            accelerator=accelerator,
                            real_npy=os.path.join(
                                args.data_dir, f"VIRTUAL_imagenet{args.resolution}_labeled.npz"
                            ),
                        )
                        if accelerator.process_index == 0:
                            metric_main.report_metric(
                                result_dict, run_dir=save_dir, snapshot_pkl=args.exp_name
                            )
                        stats_metrics.update(result_dict.results)

                logs = {name: value for name, value in stats_metrics.items()}
                accelerator.log(logs, step=global_step)
                fid_cur = logs["fid5k"]

            if (
                (global_step % args.checkpointing_steps == 0)
                or (global_step % args.latest_checkpointing_steps == 0)
            ) and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "generator": accelerator.unwrap_model(generator).state_dict(),
                        "discriminator": accelerator.unwrap_model(discriminator).state_dict(),
                        "ema": ema.state_dict(),
                        "optG": optimizerG.state_dict(),
                        "optD": optimizerD.state_dict(),
                        "args": args,
                        "steps": global_step,
                        "wallclock_time": progress_bar.format_dict["elapsed"],
                        "wandb_run_id": (
                            accelerator.get_tracker("wandb").run.id if use_wandb else None
                        ),
                    }
                    if global_step % args.checkpointing_steps == 0:
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if global_step % args.latest_checkpointing_steps == 0:
                        checkpoint_path = f"{checkpoint_dir}/latest.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if fid_best > fid_cur:
                        checkpoint_path = f"{checkpoint_dir}/fid_best.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Update best FID-5K checkpoint to {checkpoint_path}")
                        fid_best = fid_cur

            sheet_due = (
                args.sheet_steps > 0
                and global_step > 0
                and global_step % args.sheet_steps == 0
            )
            wandb_sample_due = (
                global_step > 0
                and args.sampling_steps > 0
                and global_step % args.sampling_steps == 0
            )
            if sheet_due or wandb_sample_due:
                with torch.no_grad():
                    samples = ema(x=xs_vis, z=zs_vis, y=ys_vis, truncation_psi=0.5)
                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.0

                    samples_notruc = ema(x=xs_vis, z=zs_vis, y=ys_vis, truncation_psi=0.0)
                    samples_notruc = vae.decode((samples_notruc - latents_bias) / latents_scale).sample
                    samples_notruc = (samples_notruc + 1) / 2.0

                out_samples = accelerator.gather(samples.to(torch.float32))
                out_samples_notruc = accelerator.gather(samples_notruc.to(torch.float32))

                if sheet_due and accelerator.is_main_process:
                    sheet_path = save_sample_sheet(
                        out_samples, save_dir, epoch + 1, global_step
                    )
                    save_sample_sheet(
                        out_samples_notruc,
                        save_dir,
                        epoch + 1,
                        global_step,
                        suffix="no_trunc",
                    )
                    logger.info(f"Saved sample sheets to {sheet_path}")

                if wandb_sample_due and use_wandb:
                    accelerator.log({"samples": wandb.Image(array2grid(out_samples))})
                    accelerator.log(
                        {"samples w/o trunc": wandb.Image(array2grid(out_samples_notruc))}
                    )

            logs = {}
            for loss_dict in (disc_loss_dict, gen_loss_dict):
                for k in loss_dict.keys():
                    logs[k] = accelerator.gather(loss_dict[k].mean()).mean().detach().item()

            logs["x_last_std"] = (
                accelerator.gather(accelerator.unwrap_model(generator).recent_x_std.mean()).mean().detach().item()
            )
            logs["x_last_std_disc"] = (
                accelerator.gather(accelerator.unwrap_model(discriminator).recent_x_std.mean()).mean().detach().item()
            )

            progress_bar.set_postfix(**logs)
            logs["wallclock_time"] = progress_bar.format_dict["elapsed"]
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    generator.eval()
    discriminator.eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="CAT Training")

    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=1250)
    parser.add_argument("--sheet-steps", type=int, default=0, help="Save local sample-sheet PNGs every N optimizer steps. Set 0 to disable.")
    parser.add_argument("--eval-steps", type=int, default=2500)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--wandb-name", type=str, default="CAT")

    parser.add_argument("--model", type=str, default="CAT-G-B/2")
    parser.add_argument("--modelD", type=str, default="CAT-D-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--R1_gamma", type=float, default=1.0)
    parser.add_argument("--R2_gamma", type=float, default=1.0)
    parser.add_argument("--R1_every", type=int, default=1)
    parser.add_argument("--R2_every", type=int, default=1)

    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[128, 256], default=256)
    parser.add_argument("--dit-features-dir", type=str, default=None)
    parser.add_argument("--dit-labels-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)

    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=20000)
    parser.add_argument("--latest-checkpointing-steps", type=int, default=1250)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)

    parser.add_argument("--enc-type", type=str, default="dinov2-vit-b")
    parser.add_argument("--lambda-repa", type=float, default=1.0)
    parser.add_argument("--lambda-cons", type=float, default=0.1)
    parser.add_argument(
        "--alignment-logging-steps",
        type=int,
        default=2500,
        help="Log CAT alignment diagnostics every N optimizer steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--cons-weights",
        type=str,
        default="0.3333333333333333,0.5,1.0",
    )
    parser.add_argument("--gp-eps", type=float, default=0.01)
    parser.add_argument("--gp-batch-frac", type=float, default=0.25)

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    from torch._inductor import config as ind

    ind.triton.cudagraphs = False
    main(args)

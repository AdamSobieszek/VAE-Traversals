# trainer_potential.py
import os
import os.path as osp
import sys
import time
import math
import json
import shutil
from contextlib import nullcontext
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from .recognizer import Recognizer, AntisymmetricRecognizer
from .generator_vjp import FrozenGeneratorVJP

from .aux import (
    sample_z, TrainingStatTracker, update_progress, update_stdout, sec2dhms,
    CosineScheduleWithWarmup, build_adamw, ImageLogger,
    _per_k_grad_norms,
    tb_start, dual_batch_acc_from_logits,
    entropy_from_logits, collect_wave_stats,
    tb_scalars, tb_grad_norms, tb_hists, tb_figs, tb_images, clip_accum_grads_,
    tb_path_figs,
)

DTYPE = torch.float32


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainerPotential(object):
    def __init__(self, params=None, exp_dir=None, device=torch.device("cuda"), multi_gpu=False):
        if params is None:
            raise ValueError(f"Cannot build a Trainer instance with empty params: params={params}")
        self.params = params
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda"
        self.use_mps = self.device.type == "mps"
        self.multi_gpu = bool(multi_gpu)
        self.mixed_precision = getattr(self.params, "mixed_precision", "no") or "no"
        if self.mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif self.mixed_precision == "no":
            self.amp_dtype = None
        else:
            raise ValueError(f"Unsupported mixed precision mode: {self.mixed_precision!r}")
        self.amp_enabled = self.use_cuda and self.amp_dtype is not None

        # dirs
        self.tensorboard = bool(getattr(self.params, "tensorboard", False))
        self.wip_dir = osp.join("experiments", "wip", exp_dir)
        self.complete_dir = osp.join("experiments", "complete", exp_dir)
        os.makedirs(self.wip_dir, exist_ok=True)

        self.stats_json = osp.join(self.wip_dir, "stats.json")
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, "w") as out:
                json.dump({}, out)

        self.models_dir = osp.join(self.wip_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.checkpoint = osp.join(self.models_dir, "checkpoint.pt")

        # TensorBoard (server + writer + optional ngrok)
        self.tb_writer = None
        self.tb_url = None
        self.tb = None
        self.tb_run_dir = None
        if self.tensorboard:
            self.tb_writer, self.tb_url, self.tb, self.tb_run_dir = tb_start(exp_dir)

        # loss
        self.ce_label_smoothing = float(getattr(self.params, "ce_label_smoothing", 0.00))
        self.conf_penalty_weight = float(getattr(self.params, "conf_penalty_weight", 0.0))
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.ce_label_smoothing)

        # tracker
        self.stat_tracker = TrainingStatTracker(
            ema_decay=getattr(self.params, "ema_decay", 0.9),
            ema_max_history=getattr(self.params, "ema_max_history", 200),
        )

        # filled at runtime
        self.K: Optional[int] = None
        self.T: Optional[int] = None

        # rotating image logger
        self.img_logger: Optional[ImageLogger] = None
        

    # ------------------------ small helpers ------------------------
    def _write_stats_json(self):
        with open(self.stats_json, "w") as out:
            json.dump(self.stat_tracker.stats_by_step, out)

    def log_progress(self, step_idx, mean_step_time, elapsed_time, eta):
        if step_idx > 1:
            update_stdout(12)
        stats = self.stat_tracker.stats_by_step.get(int(step_idx), {})
        total_opt_steps = math.ceil(self.params.max_iter / max(1, int(getattr(self.params, "accumulate_grad_steps", 1))))
        completed_steps = int(step_idx) + 1
        update_progress(
            "\\__.Training [bs: {}] [opt-step: {:06d}/{:06d}] ".format(
                self.params.batch_size, completed_steps, total_opt_steps
            ),
            total_opt_steps,
            completed_steps,
        )
        print()
        print("   \\__Batch accuracy Index      : {:.03f}".format(stats.get("accuracy_index", 0.0)))
        print("   \\__Step 1 accuracy           : {:.03f}".format(stats.get("step1_accuracy", 0.0)))
        print("   \\__Step 2 accuracy           : {:.03f}".format(stats.get("step2_accuracy", 0.0)))
        print("   \\__Classification loss       : {:.08f}".format(stats.get("classification_loss", 0.0)))
        print("   \\__Wave loss (PDE-JVP combo) : {:.08f}".format(
            stats.get("wave_loss", stats.get("pde_loss", 0.0))
        ))
        print("   \\__Total loss                : {:.08f}".format(stats.get("total_loss", 0.0)))
        print("      ==============================================================")
        print("   \\__Opt-step time  : {:.3f} sec".format(mean_step_time))
        print("   \\__Elapsed time   : {}".format(sec2dhms(elapsed_time)[:-6]))
        print("   \\__ETA            : {}".format(sec2dhms(eta)[:-6]))
        print("      ==============================================================")

    def _maybe_init_image_logger(self, recognizer):
        if not self.tensorboard:
            self.img_logger = None
            return  
        enable_images = bool(getattr(self.params, "enable_images", True))
        if not enable_images:
            self.img_logger = None
            return
        img_keep_last = int(getattr(self.params, "image_keep_last", 10))
        self.img_logger = ImageLogger(
            writer=self.tb_writer,
            keep_last_images=img_keep_last,
            downscale=1 / recognizer.max_pool_size if hasattr(recognizer, "max_pool_size") else 1,
        )

    def _maybe_save_initial(self, traversal_sets):
        if not osp.isfile(self.checkpoint):
            torch.save(traversal_sets.state_dict(), osp.join(self.models_dir, "traversal_sets_init.pt"))
        else:
            print("#. checkpoint found, skipping contrastive pretraining.")

    @torch.no_grad()
    def sample_dt(self, B: int, half_range: int, total_opt_steps: int) -> torch.Tensor:
        from .aux import sample_dt as sample_dt_util
        return sample_dt_util(self, B, half_range, total_opt_steps, dtype=DTYPE)

    @torch.no_grad()
    def sample_t_idx(self, B: int, target_step: int) -> torch.Tensor:
        randomize = bool(getattr(self.params, "randomize_target_step", True))
        if randomize and target_step > 1:
            return torch.randint(0, target_step-np.random.randint(0, target_step-3), (B, 1), device=self.device, dtype=torch.long)
        return torch.full((B, 1), int(target_step), device=self.device, dtype=torch.long)

    def gan_recognizer_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=True)

    def fp32_context(self):
        if not self.use_cuda:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device.type, enabled=False)

    def _generator_module(self, generator):
        return generator.module if hasattr(generator, "module") else generator

    def _maybe_prepare_generator_runtime(self, generator):
        """Prepare the generator's synthesis-only compiled runtime on-device."""
        module = self._generator_module(generator)
        if not hasattr(module, "prepare_runtime"):
            return
        if not getattr(module, "compile_enabled", False):
            return
        packed = int(self.params.batch_size) * int(self.params.num_traversal_sets)
        print(
            f"#. Compile and warm generator synthesis graphs "
            f"(mixed_precision={self.mixed_precision}, warmup batch={packed})"
        )
        warmup_z = sample_z(packed, module, self.params, self.device)
        module.prepare_runtime(warmup_z)

    # ------------------------ checkpoint utils ------------------------
    def get_starting_iteration(self, traversal_sets, recognizer,
                              traversal_opt=None, recognizer_opt=None,
                              traversal_sched=None, recognizer_sched=None):
        def safe_load_state_dict(obj, ckpt, name, strict=True):
            if obj is not None and name in ckpt:
                try:
                    obj.load_state_dict(ckpt[name], strict=strict) if "strict" in obj.load_state_dict.__code__.co_varnames else obj.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f"Error loading state_dict for {name}: {e}")

        start_iter = 0
        if osp.isfile(self.checkpoint):
            ckpt = torch.load(self.checkpoint, map_location=self.device)
            start_iter = int(ckpt.get("iter", 1))

            safe_load_state_dict(traversal_sets, ckpt, "traversal_sets", strict=False)
            safe_load_state_dict(recognizer, ckpt, "recognizer", strict=False)

            # Add noise after loading (kept as-is)
            def add_noise_to_params(module, std=1e-3):
                for p in module.parameters():
                    if p.requires_grad:
                        p.data.add_(torch.randn_like(p) * std)

            if traversal_sets is not None:
                add_noise_to_params(traversal_sets)
            if recognizer is not None:
                add_noise_to_params(recognizer)

            safe_load_state_dict(traversal_opt, ckpt, "traversal_opt")
            safe_load_state_dict(recognizer_opt, ckpt, "recognizer_opt")
            safe_load_state_dict(traversal_sched, ckpt, "traversal_sched")
            safe_load_state_dict(recognizer_sched, ckpt, "recognizer_sched")

            self.stat_tracker.set_opt_step(start_iter)

        return start_iter

    # ------------------------ optim/sched ------------------------
    def init_optimizers(self, traversal_sets, recognizer, acc_steps: int):
        traversal_set_wd = float(getattr(self.params, "traversal_set_wd", 0.01))
        recognizer_wd = float(getattr(self.params, "recognizer_wd", 0.01))
        betas = tuple(getattr(self.params, "adam_betas", (0.9, 0.999)))
        eps = float(getattr(self.params, "adam_eps", 1e-8))

        reset_lr = bool(getattr(self.params, "reset_lr", True))
        reset_weight_decay = bool(getattr(self.params, "reset_weight_decay", False))
        reset_schedulers = bool(getattr(self.params, "reset_schedulers", False))
        reset_start_iter = bool(getattr(self.params, "reset_start_iter", False))

        def force_optimizer_lr(optimizer, lr: float):
            lr = float(lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr

        def sync_scheduler_lrs(scheduler, optimizer):
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = list(lrs)
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = list(lrs)

        traversal_sets_optim = build_adamw(
            [
                {"params": traversal_sets.F.parameters(), "weight_decay": traversal_set_wd, "lr": self.params.traversal_set_lr},
                {"params": [traversal_sets.c], "weight_decay": 0.0, "lr": self.params.traversal_set_lr},
            ],
            lr=self.params.traversal_set_lr,
            weight_decay=0.0,
            extra_no_decay_names=(),
            betas=betas,
            eps=eps,
        )

        recognizer_optim = build_adamw(
            recognizer,
            lr=self.params.recognizer_lr,
            weight_decay=recognizer_wd,
            extra_no_decay_names=(),
            betas=betas,
            eps=eps,
        )

        total_opt_steps = max(1, math.ceil(self.params.max_iter / max(1, acc_steps)))
        warmup_steps = math.ceil(float(getattr(self.params, "warmup_fraction", 0.0)) * total_opt_steps)

        sched_support = CosineScheduleWithWarmup(
            traversal_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1
        )
        sched_recon = CosineScheduleWithWarmup(
            recognizer_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1
        )

        starting_opt_step = self.get_starting_iteration(
            traversal_sets, recognizer,
            traversal_opt=traversal_sets_optim, recognizer_opt=recognizer_optim,
            traversal_sched=sched_support, recognizer_sched=sched_recon,
        )

        if reset_schedulers:
            sched_support = CosineScheduleWithWarmup(
                traversal_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1
            )
            sched_recon = CosineScheduleWithWarmup(
                recognizer_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1
            )
            if reset_weight_decay and not reset_start_iter:
                starting_opt_step = 0

        if reset_lr:
            force_optimizer_lr(traversal_sets_optim, self.params.traversal_set_lr)
            force_optimizer_lr(recognizer_optim, self.params.recognizer_lr)
            sync_scheduler_lrs(sched_support, traversal_sets_optim)
            sync_scheduler_lrs(sched_recon, recognizer_optim)

        if reset_start_iter:
            starting_micro = 1
            opt_step_idx = 0
            self.stat_tracker.set_opt_step(0)
        else:
            starting_micro = starting_opt_step * acc_steps + 1
            opt_step_idx = starting_opt_step

        traversal_sets_optim.zero_grad(set_to_none=True)
        recognizer_optim.zero_grad(set_to_none=True)
        return starting_micro, opt_step_idx, traversal_sets_optim, recognizer_optim, sched_support, sched_recon

    # ------------------------ forward+loss (all K) ------------------------
    def _synthesize(self, generator, latents):
        """One generator call, including wrappers with image-shaped latents."""
        base = self._generator_module(generator)
        if getattr(base, "uses_vae_latent_shape", False) and latents.ndim == 2:
            latents = latents.reshape(-1, base.latent_channels, base.latent_size, base.latent_size)
        with self.gan_recognizer_context():
            return generator(latents)

    def _pair_logits(self, recognizer, center, endpoint):
        """Whiten once; shared centers broadcast over the traversal axis."""
        if center.shape[0] != endpoint.shape[0]:
            pairs = endpoint.unflatten(0, (center.shape[0], -1))
            reflected = (2 * center[:, None] - pairs).flatten(0, 1)
        else:
            reflected = 2 * center - endpoint
        with self.gan_recognizer_context():
            # Invoke DataParallel.forward, not a bound method of its module.
            if isinstance(recognizer, nn.DataParallel):
                if isinstance(recognizer.module, (Recognizer, AntisymmetricRecognizer)):
                    return recognizer(reflected, endpoint, antisymmetric=True)[0]
            elif hasattr(recognizer, "antisymmetric_pair_logits"):
                return recognizer.antisymmetric_pair_logits(reflected, endpoint)
            return recognizer(reflected, endpoint)[0] - recognizer(endpoint, reflected)[0]

    def _endpoint_image(self, generator, latents):
        chunk = int(getattr(self.params, "generator_recompute_chunk", 0))
        if not chunk:
            return self._synthesize(generator, latents)
        if chunk < 0 or generator.training or any(p.requires_grad for p in generator.parameters()):
            raise ValueError("Generator recomputation requires a positive chunk and a frozen eval generator.")
        if not getattr(self._generator_module(generator), "share_initial_output", False):
            raise ValueError("Generator recomputation requires deterministic, batch-independent synthesis.")
        return FrozenGeneratorVJP.apply(latents, lambda z: self._synthesize(generator, z), chunk)

    @torch.no_grad()
    def _loss_metrics(self, cls, pde, logits, z, latent1, latent2, bridge_grad, denominator):
        """Keep reductions on device and transfer the small scalar bundle once."""
        norms = bridge_grad.norm(dim=-1)
        values = {
            "total_loss": (self.params.lambda_cls * cls + self.params.lambda_pde * pde) / denominator,
            "classification_loss": cls, "pde_loss": pde,
            "entropy": entropy_from_logits(logits.float(), as_tensor=True),
            "step1_norm": (latent1 - z[:, None]).norm(dim=-1).min(),
            "step2_norm": (latent2 - latent1).norm(dim=-1).min(),
            "traversal_grad_median": norms.median(),
            "traversal_grad_max": norms.max(), "traversal_grad_min": norms.min(),
        }
        scalars = torch.stack([v.detach().float().reshape(()) for v in values.values()])
        return dict(zip(values, scalars.cpu().tolist()))

    def loss_allK(self, traversal_sets, generator, recognizer,
                  z, t_index, dt, acc_denominator: int, *, need_images: bool = False):
        """All-K loss and backward, with the original eight-value return contract.

        Train R on the historical pair first and release that graph before
        synthesizing the differentiable endpoint. Then pass the endpoint's
        cotangent through the potential rollout together with the PDE loss.
        R's BatchNorm sees the same batches, in the same order, as before.
        """
        denominator = max(1, int(acc_denominator))
        cls_scale = float(self.params.lambda_cls) / denominator
        with self.fp32_context():
            potentials, latent1, latent2, pde, _ = traversal_sets(z, t_index, dt=dt, direction=dt)
        B, K, D = latent1.shape
        targets = torch.arange(K, device=z.device).repeat(B)
        shared = getattr(self._generator_module(generator), "share_initial_output", False)
        z0 = z if shared else z[:, None].expand(B, K, D).reshape(B * K, D)
        with torch.no_grad():
            img0 = self._synthesize(generator, z0)
            img1 = self._synthesize(generator, latent1.flatten(0, 1))

        # DO_ANTISYMMETRIC_LOSS = True # TODO: Replace with global flag
        logits0 = self._pair_logits(recognizer, img0, img1)
        # if t_index == 0, then img0==img1, so we mask-out the loss from this invalid classification case
        mask = (t_index.reshape(B, 1) != 0).expand(B, K).reshape(-1)
        loss0 = nn.functional.cross_entropy(logits0.float(), targets, reduction="none")
        loss0 = (loss0 * mask).sum() / mask.sum().clamp_min(1)
        if cls_scale:
            (cls_scale * loss0).backward()
        logits0, loss0 = logits0.detach(), loss0.detach()
        del img0

        bridge = latent2.detach().flatten(0, 1).requires_grad_(True)
        img2 = self._endpoint_image(generator, bridge)
        logits = self._pair_logits(recognizer, img1, img2)
        cls = self.cross_entropy(logits.float(), targets)
        if cls_scale:
            (cls_scale * cls).backward()
            if bridge.grad is None:
                raise RuntimeError("Generator+Recognizer loss did not produce a latent gradient.")
        bridge_grad = bridge.grad if cls_scale else torch.zeros_like(bridge)

        outputs = [(latent2, bridge_grad.reshape_as(latent2)),
                   (pde, torch.full_like(pde, float(self.params.lambda_pde) / denominator))]
        outputs = [(value, cotangent) for value, cotangent in outputs if value.requires_grad]
        if outputs:
            torch.autograd.backward(*zip(*outputs))

        # Preserve legacy spatial-last gradient telemetry for VAE-shaped latents.
        base = self._generator_module(generator)
        metric_grad = (bridge_grad.reshape(-1, base.latent_channels, base.latent_size, base.latent_size)
                       if getattr(base, "uses_vae_latent_shape", False) else bridge_grad)
        metrics = self._loss_metrics(cls + loss0, pde, logits, z, latent1, latent2,
                                     metric_grad, denominator)
        images = [img.detach().unflatten(0, (B, K)) if need_images else None for img in (img1, img2)]
        return metrics, logits.detach(), logits0, targets, potentials.detach(), *images, latent2.detach()

    # ------------------------ train ------------------------
    def train(self, generator, traversal_sets, recognizer):
        # runtime toggles (defaults match your current script)
        enable_analytics = bool(getattr(self.params, "enable_analytics", True))
        enable_histograms = bool(getattr(self.params, "enable_histograms", False))
        enable_figures = bool(getattr(self.params, "enable_figures", True))
        enable_images = bool(getattr(self.params, "enable_images", True))
        save_checkpoints = bool(getattr(self.params, "save_checkpoints", True))

        self._maybe_save_initial(traversal_sets)

        # Modes / devices
        generator = generator.to(self.device).eval()
        generator.requires_grad_(False)
        traversal_sets = traversal_sets.to(self.device).train()
        recognizer = recognizer.to(self.device).train()
        for module in (generator, recognizer):
            if hasattr(module, "set_mixed_precision"):
                module.set_mixed_precision(self.mixed_precision)

        self._maybe_prepare_generator_runtime(generator)

        if self.multi_gpu:
            print(f"#. Parallelize G, R over {torch.cuda.device_count()} GPUs...")
            generator = DataParallelPassthrough(generator)
            recognizer = DataParallelPassthrough(recognizer)
            cudnn.benchmark = True

        # Bookkeeping
        self.K = int(self.params.num_traversal_sets)
        self.T = int(self.params.num_traversal_timesteps)
        self.stat_tracker.init_per_k(self.K)

        half_range = int(self.T // 2)
        target_step = int(half_range - 1)
        latent_dim_correction = 1 / (generator.dim_z.item() if isinstance(generator.dim_z, torch.Tensor) else generator.dim_z)**0.5

        init_truncation = float(getattr(self.params, "z_truncation", 1.0))
        acc_steps = max(1, int(getattr(self.params, "accumulate_grad_steps", 1)))
        VIS_IMAGE_K = min(32, int(self.K))

        total_opt_steps = max(1, math.ceil(int(self.params.max_iter) / max(1, acc_steps)))

        (starting_micro, _opt_step_idx,
         traversal_sets_optim, recognizer_optim,
         sched_support, sched_recon) = self.init_optimizers(traversal_sets, recognizer, acc_steps)

        self._maybe_init_image_logger(recognizer)

        t0 = time.time()

        if starting_micro > int(self.params.max_iter):
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns("checkpoint.pt"))
            sys.exit()

        print(f"#. Start training from micro-step {starting_micro}")
        print(f"#. Training loop: {starting_micro} to {self.params.max_iter}")
        print(f"#. Generator+Recognizer precision: {self.mixed_precision}")

        last_ckp_step = None  # avoid redundant saves
        iter_t0 = time.time()

        for micro_idx, iteration in enumerate(range(starting_micro, int(self.params.max_iter) + 1), start=1):
            # Boolean flags
            is_boundary = (micro_idx % acc_steps == 0) or (iteration == int(self.params.max_iter))
            step_idx = int(self.stat_tracker.global_opt_step)
            do_log = bool(self.tensorboard)
            do_freq = do_log and (step_idx % int(self.params.log_freq) == 0)

            do_imgs = do_log and enable_images and do_freq and (self.img_logger is not None) and is_boundary
            do_hists = do_log and enable_histograms and do_freq and is_boundary
            do_figs = do_log and enable_figures and do_freq and is_boundary
            do_gradnorm = do_freq and enable_analytics and is_boundary

            B = int(self.params.batch_size)
            z = sample_z(B, generator, self.params, self.device)

            dt = self.sample_dt(B, half_range, total_opt_steps) 
            t_idx = self.sample_t_idx(B, target_step)

            loss_dict, logits_det, logits0_det, targets, potential_preds_det, img1_bk, img2_bk, latent2_bk_det = self.loss_allK(
                traversal_sets, generator, recognizer,
                z, t_idx, dt,
                acc_denominator=acc_steps,
                need_images=do_imgs,
            )

            # ===== analytics & stats (micro) =====
            with torch.no_grad():
                batch_acc, step1_acc, step2_acc, preds_2d = dual_batch_acc_from_logits(
                    logits0_det, logits_det, B, self.K, self.device
                )

                wave_stats = {}
                if enable_analytics:
                    wave_stats = collect_wave_stats(traversal_sets, potential_preds_det)

                # per-k grad norms (only meaningful if you want it)
                per_k_gn = _per_k_grad_norms(traversal_sets) if enable_analytics else None

                self.stat_tracker.add_micro(
                    acc=batch_acc,
                    step1_accuracy=step1_acc,
                    step2_accuracy=step2_acc,
                    **loss_dict,
                    **{k: float(v) for k, v in wave_stats.items() if isinstance(v, (int, float))},
                )

                if enable_analytics:
                    preds_np = preds_2d.detach().cpu().numpy()
                    self.stat_tracker.update_all_k_after_micro(preds=preds_np, grad_norms=per_k_gn)

            # ===== recognizer optimizer step @ each micro-step =====

            if do_gradnorm:
                tb_grad_norms(self.tb_writer, step_idx, traversal_sets=traversal_sets, recognizer=recognizer)


            clip_accum_grads_(traversal_sets, micro_idx=micro_idx,acc_steps=acc_steps,
                is_boundary=is_boundary,
                mode=str(getattr(self.params, "traversal_clip_mode", "delta_prophet")),
                clip_end=float(getattr(self.params, "traversal_clip_end", 1.0)),
                clip_step=getattr(self.params, "traversal_clip_step", None),          # used by "delta" (ignored by delta_prophet)
                alpha=float(getattr(self.params, "traversal_clip_alpha", 3.0)),       # delta_prophet cap = alpha * ||delta_1||
                clip_final=getattr(self.params, "traversal_clip_final", None),        # defaults to clip_end inside util
            )
            torch.nn.utils.clip_grad_norm_(recognizer.parameters(), max_norm=2.0)

            recognizer_optim.step()
            recognizer_optim.zero_grad(set_to_none=True)
            # ===== support sets optimizer step @ boundary =====
            if is_boundary:
                torch.nn.utils.clip_grad_norm_(traversal_sets.parameters(), max_norm=2.0)

                traversal_sets_optim.step()
                traversal_sets_optim.zero_grad(set_to_none=True)

                sched_support.step()
                sched_recon.step()

                # self.params.z_truncation = init_truncation * 0.95 + 0.05 * 1.0
                self.stat_tracker.set_lrs(sched_support.get_last_lr()[0], sched_recon.get_last_lr()[0])

                win_means = self.stat_tracker.close_window()

                # TB blocks (short calls, right when inputs exist)
                if do_freq:
                    tb_scalars(self.tb_writer, step_idx, win_means, self.stat_tracker)
                    if do_hists:
                        tb_hists(
                            self.tb_writer, step_idx,
                            logits_det=logits_det,
                            potential_preds_det=potential_preds_det,
                            K=int(self.K),
                            log_potential=True,
                        )
                    if do_imgs and img1_bk is not None and img2_bk is not None:
                        tb_images(
                            self.img_logger, step_idx,
                            generator=generator,
                            z_first=z[:1],
                            img1_bk=img1_bk,
                            img2_bk=img2_bk,
                            n_vis=VIS_IMAGE_K,
                        )
                    if do_figs and enable_analytics:
                        tb_figs(self.tb_writer, step_idx, self.stat_tracker, int(self.K), int(self.params.log_freq))
                                # ===== optional figures (latent-path projections etc.) =====
                    if do_figs:
                        save_frames = bool(getattr(self.params, "save_plot_frames", True))
                        frames_dir = osp.join(self.wip_dir, "plot_frames") if save_frames else None
                        # tb_figs(self.tb_writer, step_idx, self.stat_tracker, K=int(self.K), log_freq=int(self.params.log_freq))
                        tb_path_figs(
                            self.tb_writer,
                            step_idx,
                            traversal_sets=traversal_sets,
                            z_first=z[:16],
                            dt_first=dt[:16] / dt[:16],
                            save_dir=frames_dir,
                        )
                        # Information/Confusion panel: use the same batch we already sampled (up to 64 points).
                        z_info = z[: min(int(z.shape[0]), 64)]
                        dt_info = dt[: min(int(dt.shape[0]), 64)]
                        # tb_information_matrix_figs(
                        #     self.tb_writer,
                        #     step_idx,
                        #     stat_tracker=self.stat_tracker,
                        #     traversal_sets=traversal_sets,
                        #     recognizer=recognizer,
                        #     z_bd=z_info,
                        #     dt=dt_info,
                        #     save_dir=frames_dir,
                        # )
                        # if latent2_bk_det is not None:
                        #     tb_pairwise_distance_figs(
                        #         self.tb_writer,
                        #         step_idx,
                        #         stat_tracker=self.stat_tracker,
                        #         z_bkd=latent2_bk_det,
                        #         solver=str(getattr(self.params, "pairwise_ot_solver", "sinkhorn")),
                        #         reg=float(getattr(self.params, "pairwise_ot_reg", 5e-2)),
                        #         ot_numItermax=int(getattr(self.params, "pairwise_ot_iters", 5_000)),
                        #         max_points_per_class=getattr(self.params, "pairwise_max_points_per_class", None),
                        #         save_dir=frames_dir,
                        #     )


                # timing + finalize
                step_dt = time.time() - iter_t0
                self.stat_tracker.push_step_time(step_dt)

                elapsed_time = time.time() - t0
                mean_step_time = self.stat_tracker.mean_step_time()
                eta = max(0, total_opt_steps - self.stat_tracker.global_opt_step - 1) * mean_step_time
                iter_t0 = time.time()

                self.stat_tracker.finalize_step(
                    step_idx=step_idx,
                    window_means=win_means,
                    elapsed_from_start=elapsed_time,
                    mean_step_time=mean_step_time,
                    eta_seconds=eta,
                )
                self.log_progress(step_idx, mean_step_time, elapsed_time, eta)

                if (self.stat_tracker.global_opt_step % int(self.params.log_freq)) == 0:
                    self._write_stats_json()

            # ===== checkpoint =====
            # Save once per optimizer step (boundary), not repeatedly per micro-step.
            if save_checkpoints and is_boundary:
                cur_step = int(self.stat_tracker.global_opt_step)
                if (cur_step % int(self.params.ckp_freq)) == 0 and cur_step != last_ckp_step:
                    checkpoint_dict = {
                        "iter": cur_step,
                        "traversal_sets": traversal_sets.state_dict(),
                        "recognizer": recognizer.state_dict(),
                        "traversal_opt": traversal_sets_optim.state_dict(),
                        "recognizer_opt": recognizer_optim.state_dict(),
                        "traversal_sched": sched_support.state_dict(),
                        "recognizer_sched": sched_recon.state_dict(),
                    }
                    torch.save(checkpoint_dict, self.checkpoint)
                    last_ckp_step = cur_step

        # === end loop ===
        torch.save(traversal_sets.state_dict(), osp.join(self.models_dir, "traversal_sets.pt"))
        torch.save(
            recognizer.module.state_dict() if self.multi_gpu else recognizer.state_dict(),
            osp.join(self.models_dir, "recognizer.pt"),
        )

        if self.img_logger is not None:
            self.img_logger.close()

        print("\n" * 10)
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))
        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns("checkpoint.pt"))
            print(" \\__Done!")
        except IOError as e:
            print(" \\__Already exists -- {}".format(e))

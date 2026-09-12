"""Logging, visualization, experiment output, and image-loading helpers."""

import os
import os.path as osp
import json
import numpy as np
import torch
import sys
import math
import time
from PIL import Image, ImageDraw
from typing import Optional
from pathlib import Path
from typing import Optional, Tuple
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
import re
import cv2
from torch.utils import data
import glob

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def create_exp_dir(args, new_experiment=False):
    """Create output directory for current experiment under experiments/wip/ and save given the arguments (json) and
    the given command (bash script).

    Experiment's directory name format:

        <gan_type>(-<stylegan2_resolution>)(-{Z,W})-<recognizer_type>-K<num_traversal_sets>-
            D<num_traversal_dipoles>(-LearnAlphas)(-LearnGammas)-eps<min_shift_magnitude>_<max_shift_magnitude>
    E.g.:

        experiments/wip/ProgGAN-ResNet-K200-N32-LearnGammas-eps0.35_0.5

    Args:
        args (argparse.Namespace): the namespace object returned by `parse_args()` for the current run

    """
    if new_experiment:
        print("Creating new experiment\n"+"-"*30+"\n"*2)
    exp_dir = "{}".format(args.gan_type)
    if args.gan_type == 'StyleGAN2':
        exp_dir += '-{}'.format(args.stylegan2_resolution)
        if getattr(args, "early_output", False):
            exp_dir += '-EarlyOutput'
        if args.shift_in_w_space:
            exp_dir += '-W'
        else:
            exp_dir += '-Z'
    if args.gan_type == 'BigGAN':
        biggan_classes = '-'
        for c in args.biggan_target_classes:
            biggan_classes += '{}'.format(c)
        exp_dir += '{}'.format(biggan_classes)
    exp_dir += "-{}".format(args.recognizer_type)
    exp_dir += "-K{}-D{}".format(args.num_traversal_sets, args.num_traversal_timesteps)
    if new_experiment:
        exp_dir += f"__{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        # grep all folders that start with exp_dir
        os.makedirs("experiments/wip", exist_ok=True)
        exp_dirs = [d for d in os.listdir("experiments/wip") if d.startswith(exp_dir)]
        # exclude folders that do not contain checkpoint.pt as a file in their recursive folder structure
        exp_dirs = [d for d in exp_dirs if osp.isfile(osp.join("experiments/wip", d, "models", "checkpoint.pt"))]
        # sort by last modified time
        exp_dirs.sort(key=lambda x: os.path.getmtime(osp.join("experiments/wip", x)))
        #  set exp_dir to the newest folder
        if exp_dirs:
            exp_dir = exp_dirs[-1]
        else:
            exp_dir = exp_dir + f"__{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"Using existing experiment: {exp_dir}\n"+"-"*30+"\n"*2)
    # Create output directory (wip)
    wip_dir = osp.join("experiments", "wip", exp_dir)
    os.makedirs(wip_dir, exist_ok=True)
    # Save args namespace object in json format
    with open(osp.join(wip_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Save the given command in a bash script file
    with open(osp.join(wip_dir, 'command.sh'), 'w') as command_file:
        command_file.write('#!/usr/bin/bash\n')
        command_file.write(' '.join(sys.argv) + '\n')

    return exp_dir


def module_grad_norm(mod):
    total_sq = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().to('cpu').pow(2).sum().item())
    return math.sqrt(total_sq)


@torch.no_grad()
def _per_k_grad_norms(traversal_sets) -> np.ndarray:
    K = traversal_sets.num_traversal_sets
    # Keep accumulation on the same device as gradients (prevents device sync/copies)
    try:
        dev = next(traversal_sets.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    g2 = torch.zeros(K, dtype=torch.float32, device=dev)

    for p in traversal_sets.parameters():
        g = p.grad
        if g is None:
            continue
        g = g.detach().float()
        if g.ndim == 0:
            continue
            
        # find which axis corresponds to K (don’t assume it’s dim 0)
        axes_with_K = [ax for ax, sz in enumerate(g.shape) if sz == K]
        if not axes_with_K:
            continue
        k_ax = axes_with_K[0]
        if k_ax != 0:
            g = g.movedim(k_ax, 0)  # put K in front
        g2 += g.reshape(K, -1).pow(2).sum(dim=1)
    return torch.sqrt(torch.clamp(g2, min=1e-12)).detach().cpu().numpy()


class TrainingStatTracker(object):
    """
    Tracks metrics at two levels:
      - micro-step accumulation (within a grad-acc window)
      - optimizer-step aggregates (emitted once per window)

    Also tracks per-MLP analytics (EMA accuracy, EMA grad-norm, selection counts, confusion),
    optional histories for heatmaps, and learning rates.
    """

    def __init__(self, ema_decay: float = 0.9, ema_max_history: int = 200):
        # Window (micro-step) accumulators
        self._reset_window()

        # Global optimizer-step index (used by logging)
        # Convention: this marks the CURRENT step id used for logging;
        # it is incremented AFTER finalize_step() completes.
        self.global_opt_step: int = 0

        # LRs (latest seen per optimizer-step)
        self.last_traversal_lr = 0.0
        self.last_recognizer_lr = 0.0

        # Timing
        self.iter_times = np.array([])  # seconds per opt step

        # Per-MLP analytics (set after K is known)
        self.K = None
        self.ema_decay = float(ema_decay)
        self.ema_max_history = int(ema_max_history)
        self.per_k_ema_acc = None         # [K] float
        self.per_k_ema_grad = None        # [K] float
        self.per_k_select_counts = None   # [K] long
        self.confusion = None             # [K, K] long  (row=true, col=pred)
        self.ema_history = []             # list of np.array([K]) snapshots
        self.iter_history = []            # matching optimizer-step indices for heatmap

        # JSON-like store of per-step aggregates (string keys)
        self.stats_by_step = {}  # {step_idx: dict}

        # Pairwise (class x class) distance tracking (optional; filled only when computed)
        self.pairwise_avg_dist = None   # [K,K] float
        self.pairwise_w1 = None         # [K,K] float
        self.pairwise_w2 = None         # [K,K] float
        self.pairwise_steps = []        # list[int]
        self.pairwise_w1_mean_hist = [] # list[float]
        self.pairwise_w2_mean_hist = [] # list[float]

        # Information matrix / disentanglement tracking
        self.info_lambda = None            # [K,K] float (Lambda_ij = (dot)^2)
        self.info_conf_prob = None         # [K,K] float (row-stochastic p(pred|true))
        self.info_steps = []               # list[int]
        # histories (compute both metrics for both matrices)
        self.info_Q_lambda_hist = []       # list[float]  Q(Lambda)
        self.info_Q_conf_hist = []         # list[float]  Q(mathbf p)
        self.info_deff_lambda_hist = []    # list[float]  d_eff(Lambda)
        self.info_deff_conf_hist = []      # list[float]  d_eff(mathbf p^Tmathbf p)

    # ---------- window (micro-steps) ----------
    def _reset_window(self):
        self.win_count = 0
        self.win_sum =  dict()

    def _acc(self, key: str, val: float | None):
        if val is None:  # allow optional arguments
            return
        self.win_sum[key] = self.win_sum.get(key, 0.0) + float(val)

    def add_micro(
        self,
        *,
        acc: float,
        classification_loss: float,
        total_loss: float,
        entropy: float = 0.0,
        step1_norm: float = 0.0,
        step2_norm: float = 0.0,
        potential_std: float = 0.0,
        xf_now: float = 0.0,
        # ---- allow arbitrary extras without breaking ----
        **extras,
    ):
        """Accumulate values from a micro-step; all inputs are Python floats."""
        self.win_count += 1
        self._acc('accuracy_index', acc)
        self._acc('L_classification', classification_loss)
        self._acc('total_loss', total_loss)
        self._acc('entropy', entropy)
        self._acc('step1_norm', step1_norm)
        self._acc('step2_norm', step2_norm)
        self._acc('potential_std', potential_std)
        self._acc('xf_now', xf_now)
        # PDE components

        # Any extra scalar metrics can be merged automatically
        for k, v in extras.items():
            try:
                self._acc(k, v)
            except Exception:
                # ignore non-scalar or malformed extras
                pass

    def close_window(self):
        """Return window means and reset micro accumulators."""
        denom = max(1, self.win_count)
        means = {k: (v / denom) for k, v in self.win_sum.items()}
        self._reset_window()
        return means

    # ---------- per-MLP analytics ----------
    def init_per_k(self, K: int):
        """Call once when K is known."""
        self.K = int(K)
        self.per_k_ema_acc = np.zeros(self.K, dtype=np.float32)
        self.per_k_ema_grad = np.zeros(self.K, dtype=np.float32)
        self.per_k_select_counts = np.zeros(self.K, dtype=np.int64)
        self.confusion = np.zeros((self.K, self.K), dtype=np.int64)
        self.ema_history.clear()
        self.iter_history.clear()

        self.pairwise_avg_dist = None
        self.pairwise_w1 = None
        self.pairwise_w2 = None
        self.pairwise_steps.clear()
        self.pairwise_w1_mean_hist.clear()
        self.pairwise_w2_mean_hist.clear()

        self.info_lambda = None
        self.info_conf_prob = None
        self.info_steps.clear()
        self.info_Q_lambda_hist.clear()
        self.info_Q_conf_hist.clear()
        self.info_deff_lambda_hist.clear()
        self.info_deff_conf_hist.clear()

    def update_information_metrics(
        self,
        *,
        step_idx: int,
        lambda_kxk: np.ndarray,
        conf_prob_kxk: np.ndarray,
        max_history: int = 200,
        eps: float = 1e-8,
    ):
        """
        Store latest Lambda and confusion-prob matrix, and append scalar invariants histories.

        Lambda is expected to be symmetric and nonnegative (squared dot products).
        Confusion prob is expected row-stochastic (rows sum to ~1 when counts exist).
        For invariant comparisons, we use the symmetric Gram matrix G = Σ^T Σ.
        """
        if self.K is None:
            return
        K = int(self.K)
        L = np.array(lambda_kxk, dtype=np.float64).reshape(K, K)
        S = np.array(conf_prob_kxk, dtype=np.float64).reshape(K, K)
        self.info_lambda = L.astype(np.float32)
        self.info_conf_prob = S.astype(np.float32)

        def _Q(A: np.ndarray) -> float:
            # Q(A)=det(A)^2 / det(A⊙A) with small diagonal regularization.
            A = np.array(A, dtype=np.float64, copy=False)
            A_reg = A + float(eps) * np.eye(K, dtype=np.float64)
            AA_reg = (A * A) + float(eps) * np.eye(K, dtype=np.float64)
            s1, ld1 = np.linalg.slogdet(A_reg)
            s2, ld2 = np.linalg.slogdet(AA_reg)
            if s1 <= 0 or s2 <= 0:
                return 0.0
            return float(np.exp(2.0 * ld1 - ld2))

        def _deff(A: np.ndarray) -> float:
            # d_eff(A)=(tr A)^2 / tr(A^2) where A^2 is matrix product.
            A = np.array(A, dtype=np.float64, copy=False)
            tr = float(np.trace(A))
            tr2 = float(np.trace(A @ A))
            return float((tr * tr) / max(float(eps), tr2))

        # Use a symmetric Gram matrix for confusion: G = S^T S (PSD).
        G = S.T @ S
        Q_L = _Q(L)
        Q_G = _Q(G)
        dL = _deff(L)
        dG = _deff(G)

        self.info_steps.append(int(step_idx))
        self.info_Q_lambda_hist.append(Q_L)
        self.info_Q_conf_hist.append(Q_G)
        self.info_deff_lambda_hist.append(dL)
        self.info_deff_conf_hist.append(dG)

        if len(self.info_steps) > int(max_history):
            self.info_steps = self.info_steps[-int(max_history):]
            self.info_Q_lambda_hist = self.info_Q_lambda_hist[-int(max_history):]
            self.info_Q_conf_hist = self.info_Q_conf_hist[-int(max_history):]
            self.info_deff_lambda_hist = self.info_deff_lambda_hist[-int(max_history):]
            self.info_deff_conf_hist = self.info_deff_conf_hist[-int(max_history):]

    def update_pairwise_metrics(
        self,
        *,
        step_idx: int,
        avg_dist_kxk: np.ndarray,
        w1_kxk: np.ndarray,
        w2_kxk: np.ndarray,
        max_history: int = 200,
    ):
        """
        Store the latest pairwise matrices, and append mean W1/W2 (off-diagonal) histories.
        """
        if self.K is None:
            return
        K = int(self.K)
        self.pairwise_avg_dist = np.array(avg_dist_kxk, dtype=np.float32).reshape(K, K)
        self.pairwise_w1 = np.array(w1_kxk, dtype=np.float32).reshape(K, K)
        self.pairwise_w2 = np.array(w2_kxk, dtype=np.float32).reshape(K, K)

        mask = ~np.eye(K, dtype=bool)
        w1_mean = float(self.pairwise_w1[mask].mean()) if mask.any() else float(self.pairwise_w1.mean())
        w2_mean = float(self.pairwise_w2[mask].mean()) if mask.any() else float(self.pairwise_w2.mean())
        self.pairwise_steps.append(int(step_idx))
        self.pairwise_w1_mean_hist.append(w1_mean)
        self.pairwise_w2_mean_hist.append(w2_mean)

        # cap history
        if len(self.pairwise_steps) > int(max_history):
            self.pairwise_steps = self.pairwise_steps[-int(max_history):]
            self.pairwise_w1_mean_hist = self.pairwise_w1_mean_hist[-int(max_history):]
            self.pairwise_w2_mean_hist = self.pairwise_w2_mean_hist[-int(max_history):]

    def update_per_k_after_micro(
        self,
        *,
        true_k: int,
        preds: np.ndarray,         # shape [B] int64 on CPU
        batch_size: int,
        grad_norm_selected_mlp: float | None = None,
    ):
        """
        Update EMA accuracy, selection counts, and confusion for the selected k of THIS micro-step.
        - Only the selected MLP's grad-norm is meaningful to track.
        """
        if self.K is None:
            return
        true_k = int(true_k)
        # acc for this micro-batch against the selected k
        acc = float((preds == true_k).mean())
        self.per_k_ema_acc[true_k] = self.per_k_ema_acc[true_k] * self.ema_decay + acc * (1.0 - self.ema_decay)
        self.per_k_select_counts[true_k] += int(batch_size)

        # confusion row update (count predicted classes)
        preds = np.asarray(preds).reshape(-1)
        if preds.dtype.kind not in "iu":
            preds = preds.astype(np.int64, copy=False)
        K = int(self.K)
        valid = (preds >= 0) & (preds < K)
        if not np.all(valid):
            preds = preds[valid]
        binc = np.bincount(preds, minlength=K).astype(np.int64)
        self.confusion[true_k, :] += binc

        # grad norm EMA (only for the MLP that received grads)
        if grad_norm_selected_mlp is not None:
            g = float(grad_norm_selected_mlp)
            self.per_k_ema_grad[true_k] = self.per_k_ema_grad[true_k] * self.ema_decay + g * (1.0 - self.ema_decay)

    def update_all_k_after_micro(
        self,
        *,
        preds: np.ndarray,
        grad_norms: np.ndarray | None = None,
    ):
        """Vectorized update for predictions shaped [B,K] (column k is true class k)."""
        if self.K is None:
            return
        K = int(self.K)
        preds = np.asarray(preds).reshape(-1, K).astype(np.int64, copy=False)
        valid = (preds >= 0) & (preds < K)
        true = np.broadcast_to(np.arange(K, dtype=np.int64), preds.shape)

        self.per_k_ema_acc *= self.ema_decay
        self.per_k_ema_acc += (preds == true).mean(axis=0) * (1.0 - self.ema_decay)
        self.per_k_select_counts += valid.sum(axis=0)

        flat = true[valid] * K + preds[valid]
        self.confusion += np.bincount(flat, minlength=K * K).reshape(K, K)

        if grad_norms is not None:
            grad_norms = np.asarray(grad_norms, dtype=np.float32).reshape(K)
            self.per_k_ema_grad *= self.ema_decay
            self.per_k_ema_grad += grad_norms * (1.0 - self.ema_decay)

    def snapshot_per_k_history(self, step_idx: int):
        """Keep a thin history (capped) for heatmaps."""
        if self.K is None:
            return
        self.ema_history.append(self.per_k_ema_acc.copy())
        self.iter_history.append(int(step_idx))
        if len(self.ema_history) > self.ema_max_history:
            self.ema_history = self.ema_history[-self.ema_max_history:]
            self.iter_history = self.iter_history[-self.ema_max_history:]

    # ---------- LRs ----------
    def set_lrs(self, traversal_lr: float, recognizer_lr: float):
        self.last_traversal_lr = float(traversal_lr)
        self.last_recognizer_lr = float(recognizer_lr)

    # Allow trainer to sync starting index (resume)
    def set_opt_step(self, step_idx: int):
        self.global_opt_step = int(step_idx)

    # ---------- per-step finalize ----------
    def finalize_step(
        self,
        *,
        step_idx: int,
        window_means: dict,
        elapsed_from_start: float,
        mean_step_time: float,
        eta_seconds: float,
    ):
        """
        Called once per optimizer step to store a compact dictionary of metrics.
        - Stores under the provided step_idx
        - Exposes both legacy and new metric keys for backward compatibility
        - Increments global_opt_step AFTER storing
        """
        rec = dict(window_means)

        # Backward-compat aliases expected by some logs
        if 'classification_loss' not in rec and 'L_classification' in rec:
            rec['classification_loss'] = rec['L_classification']
        if 'kl_loss' not in rec and 'L_kl' in rec:
            rec['kl_loss'] = rec['L_kl']

        rec.update({
            'traversal_sets_lr': self.last_traversal_lr,
            'recognizer_lr': self.last_recognizer_lr,
            'mean_step_time_sec': float(mean_step_time),
            'elapsed_sec': float(elapsed_from_start),
            'eta_sec': float(eta_seconds),
        })
        self.stats_by_step[int(step_idx)] = rec

        # Advance the global step *after* storing
        self.global_opt_step = int(step_idx) + 1

    # ---------- time helpers ----------
    def push_step_time(self, dt_seconds: float):
        self.iter_times = np.append(self.iter_times, float(dt_seconds))

    def mean_step_time(self) -> float:
        return float(self.iter_times.mean()) if self.iter_times.size > 0 else 0.0


def update_progress(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    block_symbol = u"\u2588"
    empty_symbol = u"\u2591"
    text = "\r{}{} {:.0f}% {}".format(msg, block_symbol * block + empty_symbol * (bar_length - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_stdout(num_lines):
    """Move cursor up and clear lines in terminal-friendly way."""
    cursor_up = '\x1b[1A'
    erase_line = '\x1b[2K'
    for _ in range(num_lines):
        sys.stdout.write(cursor_up + erase_line + '\r')
    sys.stdout.flush()


def sec2dhms(t):
    """Convert seconds to 'DD days, HH hours, MM minutes, and SS seconds'."""
    t = int(t)
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    return "%02d days, %02d hours, %02d minutes, and %02d seconds" % (day, hour, minutes, seconds)


def get_wh(img_paths):
    """Get width and height of images in given list of paths. Images are expected to have the same resolution.

    Args:
        img_paths (list): list of image paths

    Returns:
        width (int)  : the common images width
        height (int) : the common images height

    """
    img_widths = []
    img_heights = []
    for img in img_paths:
        img_ = Image.open(img)
        img_widths.append(img_.width)
        img_heights.append(img_.height)

    if len(set(img_widths)) == len(set(img_heights)) == 1:
        return img_widths[0], img_heights[1]
    else:
        raise ValueError("Inconsistent image resolutions in {}".format(img_paths))


def create_summarizing_gif(imgs_root, gif_filename, num_imgs=None, gif_size=None, gif_fps=30, gap=15, progress_bar_h=15,
                           progress_bar_color=(252, 186, 3)):
    """Create a summarizing GIF image given an images root directory (images generated across a certain latent path) and
    the number of images to appear as a static sequence. The resolution of the resulting GIF image will be
    ((num_imgs + 1) * gif_size, gif_size). That is, a static sequence of `num_imgs` images will be depicted in front of
    the animated GIF image (the latter will use all the available images in `imgs_root`).

    Args:
        imgs_root (str)            : directory of images (generated across a certain path)
        gif_filename (str)         : filename of the resulting GIF image
        num_imgs (int)             : number of images that will be used to build the static sequence before the
                                     animated part of the GIF
        gif_size (int)             : height of the GIF image (its width will be equal to (num_imgs + 1) * gif_size)
        gif_fps (int)              : GIF frames per second
        gap (int)                  : a gap between the static sequence and the animated path of the GIF
        progress_bar_h (int)       : height of the progress bar depicted to the bottom of the animated part of the GIF
                                     image. If a non-positive number is given, progress bar will be disabled.
        progress_bar_color (tuple) : color of the progress bar

    """
    # Check if given images root directory exists
    if not osp.isdir(imgs_root):
        raise NotADirectoryError("Invalid directory: {}".format(imgs_root))

    # Get all images under given root directory
    path_images = [osp.join(imgs_root, dI) for dI in os.listdir(imgs_root) if osp.isfile(osp.join(imgs_root, dI))]
    path_images.sort()

    # Set number of images to appear in the static sequence of the GIF
    num_images = len(path_images)
    if num_imgs is None:
        num_imgs = num_images
    elif num_imgs > num_images:
        num_imgs = num_images

    # Get paths of static images
    static_imgs = []
    for i in range(0, len(path_images), math.ceil(len(path_images) / num_imgs)):
        static_imgs.append(osp.join(imgs_root, '{:06}.jpg'.format(i)))
    num_imgs = len(static_imgs)

    # Get GIF image resolution
    if gif_size is not None:
        gif_w = gif_h = gif_size
    else:
        gif_w, gif_h = get_wh(static_imgs)

    # Create PIL static image
    static_img_pil = Image.new('RGB', size=(len(static_imgs) * gif_w, gif_h))
    for i in range(len(static_imgs)):
        static_img_pil.paste(Image.open(static_imgs[i]).resize((gif_w, gif_h)), (i * gif_w, 0))

    # Create PIL GIF frames
    gif_frames = []
    for i in range(len(path_images)):
        # Create new PIL frame
        gif_frame_pil = Image.new('RGB', size=((num_imgs + 1) * gif_w + gap, gif_h), color=(255, 255, 255))

        # Paste static image
        gif_frame_pil.paste(static_img_pil, (0, 0))

        # Paste current image
        gif_frame_pil.paste(Image.open(path_images[i]).resize((gif_w, gif_h)), (num_imgs * gif_w + gap, 0))

        # Draw progress bar
        if progress_bar_h > 0:
            gif_frame_pil_drawing = ImageDraw.Draw(gif_frame_pil)
            progress = (i / len(path_images)) * gif_w
            gif_frame_pil_drawing.rectangle(xy=[num_imgs * gif_w + gap, gif_h - progress_bar_h,
                                                num_imgs * gif_w + gap + progress, gif_h],
                                            fill=progress_bar_color)

        # Append to GIF frames list
        gif_frames.append(gif_frame_pil)

    # Save GIF file
    gif_frames[0].save(
        fp=gif_filename,
        append_images=gif_frames[1:],
        save_all=True,
        optimize=False,
        loop=0,
        duration=1000 // gif_fps)


class ImageViz:
    """All image-creation helpers (moved out of Trainer)."""
    _manifold_state = {}

    @staticmethod
    def to_uint01(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().cpu()
        if torch.numel(x) == 0:
            return x
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if x.min() < 0.0:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _pick_first_batch_and_K(t: torch.Tensor, k_vis: int) -> torch.Tensor:
        if t.ndim == 5:            # [B,K,C,H,W]
            return t[0, :k_vis]
        elif t.ndim == 4:          # [B,C,H,W]
            return t[:k_vis]
        elif t.ndim == 3:          # [C,H,W]
            return t.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor ndim={t.ndim} for visualization")

    @staticmethod
    def _infer_k_vis(step1_src: torch.Tensor, n_vis: int) -> int:
        if step1_src.ndim == 5:
            return min(int(n_vis), int(step1_src.shape[1]))
        elif step1_src.ndim == 4:
            return min(int(n_vis), int(step1_src.shape[0]))
        else:
            return int(n_vis)

    @staticmethod
    def _maybe_downscale(t: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
        if scale is None or abs(scale - 1.0) < 1e-6:
            return t
        return F.interpolate(t.float(), scale_factor=scale, mode="area").type_as(t)

    @classmethod
    def make_triplet_grids(
        cls,
        x0: torch.Tensor,  # step1
        x1: torch.Tensor,  # step2
        x2: torch.Tensor,  # ref
        n_vis: int = 8,
        downscale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_vis = cls._infer_k_vis(x0, n_vis)
        s1 = cls._pick_first_batch_and_K(x0, k_vis)
        s2 = cls._pick_first_batch_and_K(x1, k_vis)

        # reference shape handling
        if x2.ndim == 5:
            ref = x2[0, :k_vis]
        elif x2.ndim == 4:
            ref = x2.repeat(k_vis, 1, 1, 1) if x2.shape[0] == 1 else x2[:k_vis]
        elif x2.ndim == 3:
            ref = x2.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        elif x2.ndim == 2:
            # Allow latent-shaped ref [B,D] (e.g. identity toy generator).
            # Try to reshape it into an image using the shape of x0/x1.
            if x0.ndim == 5:
                C, H, W = int(x0.shape[2]), int(x0.shape[3]), int(x0.shape[4])
            elif x0.ndim == 4:
                C, H, W = int(x0.shape[1]), int(x0.shape[2]), int(x0.shape[3])
            else:
                raise ValueError(f"Cannot infer image shape from x0.ndim={x0.ndim} when x2 is [B,D]")

            D = int(x2.shape[1])
            if D != C * H * W:
                raise ValueError(
                    f"Reference is [B,D] with D={D}, but expected D=C*H*W={C}*{H}*{W}={C*H*W}. "
                    "Pass an image-shaped reference, or make generator(z) return [B,C,H,W]."
                )
            x2_img = x2.view(x2.shape[0], C, H, W)
            ref = x2_img.repeat(k_vis, 1, 1, 1) if x2_img.shape[0] == 1 else x2_img[:k_vis]
        else:
            raise ValueError(f"Unexpected reference tensor ndim={x2.ndim}")

        # optional downscale
        ref = cls._maybe_downscale(ref, downscale)
        s1  = cls._maybe_downscale(s1,  downscale)
        s2  = cls._maybe_downscale(s2,  downscale)

        ref_n = cls.to_uint01(ref)
        s1_n  = cls.to_uint01(s1)
        s2_n  = cls.to_uint01(s2)

        grid_ref = make_grid(ref_n, nrow=k_vis)
        grid_s1  = make_grid(s1_n,  nrow=k_vis)
        grid_s2  = make_grid(s2_n,  nrow=k_vis)
        stacked  = torch.cat([grid_ref, grid_s1, grid_s2], dim=1)

        diff1 = (s1_n - ref_n)
        diff2 = (s2_n - s1_n)
        grid_d1 = make_grid(diff1, nrow=k_vis)
        grid_d2 = make_grid(diff2, nrow=k_vis)
        stacked_diff = torch.cat([grid_d1, grid_d2], dim=1)
        return stacked, stacked_diff

    @staticmethod
    def plot_heatmap(mat_t_by_k, K: int, title: str, xlabel: str, ylabel: str):
        arr = np.array(mat_t_by_k)
        if arr.ndim == 2 and arr.shape[0] != K:
            arr = arr.T
        fig, ax = plt.subplots(figsize=(max(6, arr.shape[1] * 0.15), max(4, K * 0.15)))
        im = ax.imshow(arr, aspect='auto', origin='lower', interpolation='nearest')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_yticks(np.arange(K)); ax.set_yticklabels([str(i) for i in range(K)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Figure saving helpers
    # ----------------------------
    @staticmethod
    def save_fig_copy(fig, *, out_dir: str | Path, tag: str, step: int, dpi: int = 140) -> str:
        """
        Save a deterministic PNG copy of a Matplotlib figure for later collation (e.g., GIF).
        Returns the written path as a string.
        """
        out_dir = Path(out_dir) / str(tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{int(step):08d}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        return str(path)

    @staticmethod
    def frames_to_gif(
        *,
        frames_dir: str | Path,
        out_gif: str | Path,
        fps: float = 6.0,
        glob_pattern: str = "*.png",
        resize: tuple[int, int] | None = None,
    ) -> str:
        """
        Collate PNG frames in `frames_dir` into a GIF.
        Frames are sorted by the leading integer in the filename, falling back to lexicographic.
        """
        if imageio is None:
            raise ImportError("imageio is required to write GIFs; install `imageio`.")

        frames_dir = Path(frames_dir)
        out_gif = Path(out_gif)
        files = sorted(frames_dir.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(f"No frames found in {frames_dir} with glob {glob_pattern!r}")

        def key(p: Path):
            m = re.match(r"^(\d+)", p.stem)
            return (int(m.group(1)) if m else 10**18, p.name)

        files = sorted(files, key=key)
        duration = 1.0 / max(1e-6, float(fps))

        imgs = []
        for f in files:
            im = imageio.imread(f)
            if resize is not None:
                # simple nearest-neighbor resize via PIL if available
                try:
                    from PIL import Image
                    im = np.array(Image.fromarray(im).resize(resize, resample=Image.Resampling.BILINEAR))
                except Exception:
                    pass
            imgs.append(im)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_gif, imgs, duration=duration, loop=0)
        return str(out_gif)

    @staticmethod
    def plot_confusion(conf_mat_nd: np.ndarray, K: int):
        cm = torch.tensor(conf_mat_nd, dtype=torch.float32)
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        cm_norm = (cm / row_sums).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, interpolation='nearest', aspect='auto', origin='lower')
        ax.set_title("Classifier Confusion (row=true k, col=pred k)")
        ax.set_xlabel("predicted k"); ax.set_ylabel("true k")
        ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Pairwise distances + OT (class x class)
    # ----------------------------
    @staticmethod
    @torch.no_grad()
    def compute_pairwise_avg_and_wasserstein(
        z_bkd: torch.Tensor,
        *,
        solver: str = "sinkhorn",         # "sinkhorn" | "emd"
        reg: float = 3e-2,               # sinkhorn regularization
        ot_numItermax: int = 7_000,
        max_points_per_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-class pairwise:
        - avg L2 distance between samples (mean of all pairwise distances)
        - Wasserstein-1 distance (OT cost with L2 ground metric)
        - Wasserstein-2 distance (sqrt of OT cost with squared L2 ground metric)

        Args:
            z_bkd: [B,K,D] latent samples per class k
        Returns:
            avg_dist[K,K], w1[K,K], w2[K,K] as float32 numpy arrays
        """
        if z_bkd.dim() != 3:
            raise ValueError(f"Expected z_bkd [B,K,D], got {tuple(z_bkd.shape)}")
        B, K, D = map(int, z_bkd.shape)

        # optional subsampling for speed
        if max_points_per_class is not None and B > int(max_points_per_class):
            idx = torch.randperm(B, device=z_bkd.device)[: int(max_points_per_class)]
            z_bkd = z_bkd.index_select(0, idx)
            B = int(z_bkd.shape[0])

        # POT is CPU/numpy oriented; keep it local-imported
        try:
            import ot  # POT
        except Exception as e:
            raise ImportError("POT is required for OT distances; install `POT`.") from e

        z = z_bkd.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)   # [B,K,D]
        a = np.full((B,), 1.0 / float(B), dtype=np.float64)

        avg = np.zeros((K, K), dtype=np.float64)
        w1 = np.zeros((K, K), dtype=np.float64)
        w2 = np.zeros((K, K), dtype=np.float64)

        solver = str(solver).lower().strip()
        if solver not in {"sinkhorn", "emd"}:
            raise ValueError("solver must be one of {'sinkhorn','emd'}")

        for i in range(K):
            Xi = z[:, i, :]  # [B,D]
            for j in range(i, K):
                if i == j:
                    continue
                Xj = z[:, j, :]
                # cost matrices
                diff = Xi[:, None, :] - Xj[None, :, :]
                C2 = np.sum(diff * diff, axis=-1)              # squared L2
                C1 = np.sqrt(np.maximum(C2, 0.0))              # L2

                avg_ij = float(np.mean(C1))

                if solver == "emd":
                    w1_cost = float(ot.emd2(a, a, C1, numItermax=int(ot_numItermax)))
                    w2_cost = float(ot.emd2(a, a, C2, numItermax=int(ot_numItermax)))
                else:
                    w1_cost = float(ot.sinkhorn2(a, a, C1, reg=float(reg), numItermax=int(ot_numItermax)))
                    w2_cost = float(ot.sinkhorn2(a, a, C2, reg=float(reg), numItermax=int(ot_numItermax)))

                w2_val = math.sqrt(max(0.0, w2_cost))

                avg[i, j] = avg[j, i] = avg_ij
                w1[i, j] = w1[j, i] = w1_cost
                w2[i, j] = w2[j, i] = w2_val

        return avg.astype(np.float32), w1.astype(np.float32), w2.astype(np.float32)

    @staticmethod
    def plot_pairwise_distance_ot_panel(
        *,
        avg_dist_kxk: np.ndarray,
        w2_kxk: np.ndarray,
        steps: list[int] | np.ndarray,
        w1_mean_hist: list[float] | np.ndarray,
        w2_mean_hist: list[float] | np.ndarray,
        title: str = "Pairwise class distances (avg) + OT (W2) with mean W1/W2 histories",
    ):
        """
        2x2 panel:
        - (0,0) heatmap: average pairwise L2 distance
        - (0,1) heatmap: Wasserstein-2 distance
        - (1,0) line: mean Wasserstein-1 over time
        - (1,1) line: mean Wasserstein-2 over time
        """
        avg = np.array(avg_dist_kxk, dtype=np.float32)
        w2 = np.array(w2_kxk, dtype=np.float32)
        if avg.ndim != 2 or avg.shape[0] != avg.shape[1]:
            raise ValueError(f"avg_dist_kxk must be square [K,K], got {avg.shape}")
        if w2.shape != avg.shape:
            raise ValueError(f"w2_kxk must match avg_dist_kxk, got {w2.shape} vs {avg.shape}")
        K = int(avg.shape[0])

        # Keep the heatmap row taller (square panels) and make the time-series row flatter.
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 11),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax00, ax01 = axs[0, 0], axs[0, 1]
        ax10, ax11 = axs[1, 0], axs[1, 1]

        im0 = ax00.imshow(avg, interpolation="nearest", aspect="auto", origin="lower")
        ax00.set_title("Avg L2 distance (between class samples)")
        ax00.set_xlabel("class j"); ax00.set_ylabel("class i")
        ax00.set_xticks(np.arange(K)); ax00.set_yticks(np.arange(K))
        fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
        ax00.set_box_aspect(1)

        im1 = ax01.imshow(w2, interpolation="nearest", aspect="auto", origin="lower")
        ax01.set_title("OT distance (W2)")
        ax01.set_xlabel("class j"); ax01.set_ylabel("class i")
        ax01.set_xticks(np.arange(K)); ax01.set_yticks(np.arange(K))
        fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
        ax01.set_box_aspect(1)

        xs = np.array(steps, dtype=np.int64).reshape(-1)
        y_w1 = np.array(w1_mean_hist, dtype=np.float32).reshape(-1)
        y_w2 = np.array(w2_mean_hist, dtype=np.float32).reshape(-1)

        ax10.plot(xs, y_w1, linewidth=1.8)
        ax10.set_title("Mean Wasserstein-1 (off-diagonal) vs step")
        ax10.set_xlabel("opt step"); ax10.set_ylabel("mean W1")
        ax10.grid(True, linewidth=0.3, alpha=0.4)

        ax11.plot(xs, y_w2, linewidth=1.8)
        ax11.set_title("Mean Wasserstein-2 (off-diagonal) vs step")
        ax11.set_xlabel("opt step"); ax11.set_ylabel("mean W2")
        ax11.grid(True, linewidth=0.3, alpha=0.4)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Bayesian/free-energy style panel (KL + KL/W2)
    # ----------------------------
    @staticmethod
    def plot_pairwise_kl_free_energy_panel(
        *,
        kl_kxk: np.ndarray,
        w2_kxk: np.ndarray,
        steps: list[int] | np.ndarray,
        kl_mean_hist: list[float] | np.ndarray,
        ratio_mean_hist: list[float] | np.ndarray,
        eps: float = 1e-8,
        title: str = "Pairwise KL (entropic) + KL/W2 ratio with mean histories",
    ):
        """
        2x2 panel:
        - (0,0) heatmap: KL_eps between entropically smoothed point clouds
        - (0,1) heatmap: KL_eps / W2_eps ratio
        - (1,0) line: mean KL_eps over time
        - (1,1) line: mean KL_eps / W2_eps over time
        """
        kl = np.array(kl_kxk, dtype=np.float32)
        w2 = np.array(w2_kxk, dtype=np.float32)
        if kl.ndim != 2 or kl.shape[0] != kl.shape[1]:
            raise ValueError(f"kl_kxk must be square [K,K], got {kl.shape}")
        if w2.shape != kl.shape:
            raise ValueError(f"w2_kxk must match kl_kxk, got {w2.shape} vs {kl.shape}")
        K = int(kl.shape[0])

        denom = np.maximum(w2, float(eps))
        ratio = kl / denom

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 4]},
        )
        ax00, ax01 = axs[0, 0], axs[0, 1]
        ax10, ax11 = axs[1, 0], axs[1, 1]

        im0 = ax00.imshow(kl, interpolation="nearest", aspect="auto", origin="lower")
        ax00.set_title("KL_eps (entropic)")
        ax00.set_xlabel("class j"); ax00.set_ylabel("class i")
        ax00.set_xticks(np.arange(K)); ax00.set_yticks(np.arange(K))
        fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
        ax00.set_box_aspect(1)

        im1 = ax01.imshow(ratio, interpolation="nearest", aspect="auto", origin="lower")
        ax01.set_title("KL_eps / W2_eps")
        ax01.set_xlabel("class j"); ax01.set_ylabel("class i")
        ax01.set_xticks(np.arange(K)); ax01.set_yticks(np.arange(K))
        fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
        ax01.set_box_aspect(1)

        xs = np.array(steps, dtype=np.int64).reshape(-1)
        y_kl = np.array(kl_mean_hist, dtype=np.float32).reshape(-1)
        y_ratio = np.array(ratio_mean_hist, dtype=np.float32).reshape(-1)

        ax10.plot(xs, y_kl, linewidth=1.8)
        ax10.set_title("Mean KL_eps (off-diagonal) vs step")
        ax10.set_xlabel("opt step"); ax10.set_ylabel("mean KL")
        ax10.grid(True, linewidth=0.3, alpha=0.4)

        ax11.plot(xs, y_ratio, linewidth=1.8)
        ax11.set_title("Mean KL_eps / W2_eps (off-diagonal) vs step")
        ax11.set_xlabel("opt step"); ax11.set_ylabel("mean ratio")
        ax11.grid(True, linewidth=0.3, alpha=0.4)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Information matrix + confusion panel
    # ----------------------------
    @staticmethod
    def plot_information_matrix_panel(
        *,
        lambda_kxk: np.ndarray,
        conf_prob_kxk: np.ndarray,
        steps: list[int] | np.ndarray,
        Q_lambda_hist: list[float] | np.ndarray,
        Q_conf_hist: list[float] | np.ndarray,
        deff_lambda_hist: list[float] | np.ndarray,
        deff_conf_hist: list[float] | np.ndarray,
        title: str = "Information matrix Λ (squared dot products) + confusion with invariants",
    ):
        """
        2x2 panel (wider for big heatmaps):
        - (0,0) heatmap: Λ_ij = (v_i · v_j)^2
        - (0,1) heatmap: row-normalized confusion p(pred|true)
        - (1,0) line: Q(Λ) = det(Λ)^2 / det(Λ ⊙ Λ)
        - (1,1) line: d_eff(Λ) and d_eff(G) with G = Σ^T Σ
        """
        L = np.array(lambda_kxk, dtype=np.float32)
        C = np.array(conf_prob_kxk, dtype=np.float32)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(f"lambda_kxk must be square [K,K], got {L.shape}")
        if C.shape != L.shape:
            raise ValueError(f"conf_prob_kxk must match lambda_kxk, got {C.shape} vs {L.shape}")
        K = int(L.shape[0])

        # Set up gridspec with 2x2, upper row (row 0) subplots square, lower row (row 1) shorter
        from matplotlib import gridspec

        fig = plt.figure(figsize=(12, 8))
        # GridSpec: make top row height much bigger than bottom
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.38], figure=fig)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])

        # Ensure the upper row subplots are square
        ax00.set_box_aspect(1)
        ax01.set_box_aspect(1)
        # Lower row can be non-square ("shorter")

        # ticks: avoid unreadable plots when K is large
        stride = 1 if K <= 32 else max(1, K // 16)
        ticks = np.arange(0, K, stride, dtype=np.int64)

        im0 = ax00.imshow(L, interpolation="nearest", aspect="auto", origin="upper")
        ax00.set_title(r"$\Lambda_{ij} = (v_i \cdot v_j)^2$")
        ax00.set_xlabel("j (label)"); ax00.set_ylabel("i (label)")
        ax00.set_xticks(ticks); ax00.set_yticks(ticks)
        fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
        ax00.set_box_aspect(1)

        im1 = ax01.imshow(C, interpolation="nearest", aspect="auto", origin="upper", vmin=0.0, vmax=1.0)
        ax01.set_title(r"$p(\mathrm{pred}\mid \mathrm{true})$ (row-normalized)")
        ax01.set_xlabel("pred"); ax01.set_ylabel("true")
        ax01.set_xticks(ticks); ax01.set_yticks(ticks)
        fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
        ax01.set_box_aspect(1)

        xs = np.array(steps, dtype=np.int64).reshape(-1)
        yQ_L = np.array(Q_lambda_hist, dtype=np.float64).reshape(-1)
        yQ_S = np.array(Q_conf_hist, dtype=np.float64).reshape(-1)
        yD_L = np.array(deff_lambda_hist, dtype=np.float64).reshape(-1)
        yD_S = np.array(deff_conf_hist, dtype=np.float64).reshape(-1)

        ax10.plot(xs, yQ_L, linewidth=1.8, label="Q(Λ)")
        ax10.plot(xs, yQ_S, linewidth=1.8, label="Q(ΣᵀΣ)")
        ax10.set_title(r"$\mathcal{Q}(\Lambda)=\det(\Lambda)^2/\det(\Lambda\odot\Lambda)$")
        ax10.set_xlabel("opt step"); ax10.set_ylabel("Q(Λ)")
        ax10.grid(True, linewidth=0.3, alpha=0.4)
        ax10.legend(loc="best", fontsize=9, frameon=False)

        # Normalize effective dimension by ambient dimension D (here: matrix dimension K) so the scale is in [0,1].
        D_dim = float(K)
        yD_Ln = yD_L / max(1.0, D_dim)
        yD_Sn = yD_S / max(1.0, D_dim)
        ax11.plot(xs, yD_Ln, linewidth=1.8, label=r"$\frac{1}{D}d_{\mathrm{eff}}(\Lambda)$")
        ax11.plot(xs, yD_Sn, linewidth=1.8, label=r"$\frac{1}{D}d_{\mathrm{eff}}(\mathbf{p}^T\mathbf{p})$")
        ax11.set_title(r"Effective dimension/dimension: $\frac{1}{D}d_{\mathrm{eff}}(A)=(\mathrm{tr}\,A)^2/\mathrm{tr}(A^2)$")
        ax11.set_xlabel("opt step"); ax11.set_ylabel(r"$\frac{1}{D}d_{\mathrm{eff}}(\mathbf{p}^T\mathbf{p})$")
        ax11.grid(True, linewidth=0.3, alpha=0.4)
        ax11.set_ylim(0.0, 1.0)
        ax11.legend(loc="best", fontsize=9, frameon=False)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    @staticmethod
    @torch.no_grad()
    def compute_lambda_and_confusion(
        *,
        traversal_sets,
        recognizer,
        z_bd: torch.Tensor,   # [B,D]
        dt: torch.Tensor,     # [B,1] or [B,K]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute:
          - Λ_ij = E_b[(û_{b,i} · û_{b,j})^2] from per-class step vectors (squared cosine similarity)
          - Σ = p(pred|true) row-normalized confusion, using the recognizer on the same uv transform as training.
        """
        if z_bd.dim() != 2:
            raise ValueError(f"Expected z_bd [B,D], got {tuple(z_bd.shape)}")
        B, D = z_bd.shape
        K = int(getattr(traversal_sets, "num_traversal_sets", 1))

        # One-step per-k displacement field
        z_bkd, delta_bkd = traversal_sets.inference(z_bd, dt=dt, return_all=False)  # [B,K,D], [B,K,D]

        # ---- Lambda: average squared dot products between normalized deltas ----
        u = delta_bkd / delta_bkd.norm(dim=-1, keepdim=True).clamp_min_(1e-12)  # [B,K,D]
        dots_bkk = torch.einsum("bkd,bjd->bkj", u, u)  # [B,K,K]
        lam = (dots_bkk ** 2).mean(dim=0).to(dtype=torch.float32).detach().cpu().numpy()  # [K,K]

        # ---- Confusion: use training-time uv(a,b) = (2a-b, b) ----
        a = z_bkd.reshape(B * K, D)
        b = (z_bkd + delta_bkd).reshape(B * K, D)
        u_in = 2.0 * a - b
        v_in = b
        logits, _ = recognizer(u_in, v_in)  # [B*K,K]
        preds = logits.argmax(dim=-1)          # [B*K]
        targets = torch.arange(K, device=preds.device).repeat(B)  # [B*K]

        # Guard against logits with extra classes: drop out-of-range preds.
        valid = (preds >= 0) & (preds < K)
        if not torch.all(valid):
            preds = preds[valid]
            targets = targets[valid]

        if preds.numel() == 0:
            counts = torch.zeros((K, K), device=logits.device, dtype=torch.float32)
        else:
            idx = targets * K + preds
            counts = torch.bincount(idx, minlength=K * K).reshape(K, K).to(dtype=torch.float32)
        row_sums = counts.sum(dim=1, keepdim=True).clamp_min_(1.0)
        conf = (counts / row_sums).detach().cpu().numpy()  # [K,K]
        return lam, conf


class ImageLogger:
    """
    Writes images into the SAME TensorBoard run directory as your main SummaryWriter.
    PNG encoding overlaps training, with at most one pending triplet. Device
    transfer and grid construction stay on the caller; the worker gets detached
    host grids only. close() drains pending images and propagates worker errors.
    Legacy image-file pruning helpers remain available but are not enabled.
    """
    def __init__(self, writer: SummaryWriter, keep_last_images: int = 50,
                 downscale: Optional[float] = None, *, asynchronous: bool = True):
        self.writer = writer
        self.keep_last_images = int(keep_last_images)
        self.downscale = downscale
        # Ensure we can glob files reliably regardless of SummaryWriter implementation.
        self._log_dir = Path(getattr(writer, "log_dir", ""))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tb-images") if asynchronous else None
        self._pending = None

    def _list_image_eventfiles(self):
        # PyTorch appends filename_suffix to event filename, so match *.images*
        return sorted(
            [p for p in self._log_dir.glob("events.out.tfevents.*.images*") if p.is_file()],
            key=lambda p: p.stat().st_mtime
        )

    def _prune_old_images(self):
        files = self._list_image_eventfiles()
        if self.keep_last_images <= 0:
            to_delete = files  # keep none
        else:
            to_delete = files[:-self.keep_last_images]
        for f in to_delete:
            try:
                f.unlink()
            except Exception:
                pass

    def log_triplet(self, tag_prefix: str, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                    step: int, n_vis: int = 8):
        # Drain before making new grids, bounding host memory to one image batch.
        self._drain()
        triplet, diffs = ImageViz.make_triplet_grids(x0, x1, x2, n_vis=n_vis, downscale=self.downscale)
        if self._executor is None:
            self._write_triplet(tag_prefix, triplet, diffs, step)
        else:
            self._pending = self._executor.submit(self._write_triplet, tag_prefix, triplet, diffs, step)

    def _write_triplet(self, tag_prefix, triplet, diffs, step):
        self.writer.add_image(f"{tag_prefix}/triplet", triplet, step)
        self.writer.add_image(f"{tag_prefix}/diff_triplet_abs", diffs, step)
        #  writer.close()
        # self._prune_old_images()
        
    def _drain(self):
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def flush(self):
        self._drain()
        self.writer.flush()

    def close(self):
        try:
            self.flush()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self.writer.close()


def tb_start(exp_dir: str):
    """
    Starts TensorBoard programmatically and optionally opens an ngrok tunnel.
    Uses env vars:
      TB_HOST, TB_PORT,
      NGROK_AUTHTOKEN, NGROK_DOMAIN, NGROK_BASIC_AUTH, NGROK_REGION
    Returns: (tb_writer, tb_url, tb_obj, run_logdir)
    """
    from tensorboard import program
    from torch.utils.tensorboard import SummaryWriter
    import os

    exp_root, run_name = exp_dir.split("__")
    tb_dir = os.path.join("experiments", "tensorboard", "wip", exp_root)
    run_dir = os.path.join(tb_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_host = os.getenv("TB_HOST", "0.0.0.0")
    tb_port = int(os.getenv("TB_PORT", "6006"))

    tb = program.TensorBoard()
    tb.configure(argv=[
        None,
        "--logdir", tb_dir,
        "--host", tb_host,
        "--port", str(tb_port),
        "--reload_interval", "5",
    ])
    local_url = tb.launch()

    public_url = None
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_token:
        try:
            from pyngrok import ngrok, conf
            cfg = conf.PyngrokConfig(auth_token=ngrok_token)
            conf.set_default(cfg)

            # Close any old tunnel on this port (useful for restarts)
            for t in ngrok.get_tunnels():
                if t.config.get("addr", "").endswith(f":{tb_port}"):
                    ngrok.disconnect(t.public_url)

            ngrok_hostname = os.getenv("NGROK_DOMAIN")   # reserved domain (premium)
            ngrok_auth = os.getenv("NGROK_BASIC_AUTH")   # "user:pass"
            ngrok_region = os.getenv("NGROK_REGION")     # e.g. "eu", "us"
            if ngrok_region:
                cfg.region = ngrok_region

            connect_kwargs = {"proto": "http", "addr": tb_port}
            if ngrok_hostname:
                connect_kwargs["hostname"] = ngrok_hostname
            if ngrok_auth:
                connect_kwargs["auth"] = ngrok_auth

            tunnel = ngrok.connect(**connect_kwargs)
            public_url = tunnel.public_url
        except Exception as e:
            print(f"[ngrok] Failed to create tunnel: {e}")

    tb_url = public_url or local_url
    print(f"#. TensorBoard local: {local_url}")
    if public_url:
        print(f"#. TensorBoard public: {public_url}", "\n" * 8)
    else:
        print("#. (No ngrok tunnel; set NGROK_AUTHTOKEN to expose publicly)")

    writer = SummaryWriter(log_dir=run_dir)
    return writer, tb_url, tb, run_dir


@torch.no_grad()
def twostep_batch_acc_from_logits(
    logits0: torch.Tensor,
    logits: torch.Tensor,
    B: int,
    K: int,
    device,
) -> tuple[float, float, float, torch.Tensor]:
    true_2d = torch.arange(K, device=device).unsqueeze(0)
    preds0 = torch.argmax(logits0, dim=1).view(B, K)
    preds = torch.argmax(logits, dim=1).view(B, K)
    accs = torch.stack(((preds0 == true_2d).float().mean(), (preds == true_2d).float().mean()))
    step1_acc, step2_acc = accs.cpu().tolist()
    acc = 0.5 * (step1_acc + step2_acc)
    return acc, step1_acc, step2_acc, preds


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor, *, as_tensor: bool = False):
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
    return ent if as_tensor else float(ent.item())


@torch.no_grad()
def collect_wave_stats(traversal_sets, potential_preds: torch.Tensor) -> dict:
    wave_dict = dict(traversal_sets.get_losses())
    wave_dict["potential_std"] = potential_preds.std()
    if "xf_now" in wave_dict and torch.is_tensor(wave_dict["xf_now"]):
        wave_dict["xf_now"] = wave_dict["xf_now"].norm(dim=-1).mean()
    # Normalize scalar tensors with one device-to-host synchronization.
    out = {}
    scalar_keys = []
    scalar_tensors = []
    for k, v in wave_dict.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                scalar_keys.append(k)
                scalar_tensors.append(v.detach().float().reshape(()))
            else:
                out[k] = v
        else:
            out[k] = float(v) if isinstance(v, (int, float)) else v
    if scalar_tensors:
        out.update(zip(scalar_keys, torch.stack(scalar_tensors).cpu().tolist()))
    return out


def _tb_finite_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tb_finite_tensor(t: torch.Tensor) -> torch.Tensor | None:
    if t is None:
        return None
    t = t.detach().float().cpu().reshape(-1)
    finite = torch.isfinite(t)
    if not finite.any():
        return None
    return t[finite]


def tb_scalars(writer, step: int, win_means: dict, stat_tracker):
    for k, v in win_means.items():
        v = _tb_finite_float(v)
        if v is not None:
            writer.add_scalar(f"train/{k}", v, step)
    traversal_lr = _tb_finite_float(stat_tracker.last_traversal_lr)
    recognizer_lr = _tb_finite_float(stat_tracker.last_recognizer_lr)
    if traversal_lr is not None:
        writer.add_scalar("train/traversal_sets_lr", traversal_lr, step)
    if recognizer_lr is not None:
        writer.add_scalar("train/recognizer_lr", recognizer_lr, step)


def tb_grad_norms(writer, step: int, traversal_sets=None, recognizer=None, freq: int = 1):
    if step % freq != 0:
        return
    if traversal_sets is not None:
        gn_support = module_grad_norm(traversal_sets.F)
        gn_support = _tb_finite_float(gn_support)
        if gn_support is not None:
            writer.add_scalar("train/grad_norm/traversal_sets", gn_support, step)
    if recognizer is not None:
        gn_recon = module_grad_norm(recognizer)
        gn_recon = _tb_finite_float(gn_recon)
        if gn_recon is not None:
            writer.add_scalar("train/grad_norm/recognizer", gn_recon, step)


def tb_hists(writer, step: int, *, logits_det: torch.Tensor,
             potential_preds_det: torch.Tensor | None,
             K: int, log_potential: bool = True):
    logits_tb = _tb_finite_tensor(logits_det)
    if logits_tb is not None:
        writer.add_histogram("train/logits", logits_tb, step)
    if log_potential and potential_preds_det is not None:
        potential_cpu = potential_preds_det.detach().float().cpu()
        for k in range(K):
            potential_tb = potential_cpu[:, k].reshape(-1)
            potential_tb = potential_tb[torch.isfinite(potential_tb)]
            if potential_tb.numel():
                writer.add_histogram(f"potential_distribution/{k}", potential_tb, step)


def tb_figs(writer, step: int, stat_tracker, K: int, log_freq: int):
    # snapshot history only when we're logging figures
    stat_tracker.snapshot_per_k_history(step)

    if len(stat_tracker.ema_history) >= 2:
        hist_mat = np.stack(stat_tracker.ema_history, axis=1)[:, -30:]
        fig = ImageViz.plot_heatmap(
            hist_mat, K=K,
            title="Per-MLP EMA Accuracy over Time",
            xlabel="optimizer step snapshot",
            ylabel="MLP index k",
        )
        writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=step)
        plt.close(fig)

        fig_c = ImageViz.plot_confusion(stat_tracker.confusion, K=K)
        writer.add_figure("classifier/confusion_matrix", fig_c, global_step=step)
        plt.close(fig_c)


@torch.no_grad()
def decode_generator_output_for_viz(generator, output):
    """Decode latent generator outputs to RGB for TensorBoard / image logging."""
    base = generator.module if hasattr(generator, "module") else generator
    decode = getattr(base, "decode_with_vae", None)
    if decode is None:
        return output
    leading = output.shape[:-3]
    flat = output.reshape(-1, *output.shape[-3:])
    images = decode(flat)
    return images.reshape(*leading, *images.shape[1:])


@torch.no_grad()
def tb_images(img_logger, step: int, *, generator, z_first: torch.Tensor,
              img1_bk: torch.Tensor, img2_bk: torch.Tensor, n_vis: int):
    first_img = decode_generator_output_for_viz(generator, generator(z_first))
    img1_viz = decode_generator_output_for_viz(generator, img1_bk)
    img2_viz = decode_generator_output_for_viz(generator, img2_bk)
    img_logger.log_triplet(
        tag_prefix="images",
        x0=img1_viz, x1=img2_viz, x2=first_img,
        step=step,
        n_vis=n_vis,
    )


def tb_pairwise_distance_figs(
    writer,
    step: int,
    *,
    stat_tracker,
    z_bkd: torch.Tensor,
    solver: str = "sinkhorn",
    reg: float = 2e-2,
    ot_numItermax: int = 5_000,
    max_points_per_class: int | None = None,
    save_dir: str | Path | None = None,
):
    """
    Compute + log a 2x2 panel of:
      - avg pairwise distance heatmap
      - OT/W2 heatmap
      - mean W1 history
      - mean W2 history

    Expects z_bkd shaped [B,K,D].
    """
    avg, w1, w2 = ImageViz.compute_pairwise_avg_and_wasserstein(
        z_bkd,
        solver=solver,
        reg=reg,
        ot_numItermax=ot_numItermax,
        max_points_per_class=max_points_per_class,
    )
    stat_tracker.update_pairwise_metrics(step_idx=int(step), avg_dist_kxk=avg, w1_kxk=w1, w2_kxk=w2)
    fig = ImageViz.plot_pairwise_distance_ot_panel(
        avg_dist_kxk=avg,
        w2_kxk=w2,
        steps=stat_tracker.pairwise_steps,
        w1_mean_hist=stat_tracker.pairwise_w1_mean_hist,
        w2_mean_hist=stat_tracker.pairwise_w2_mean_hist,
    )
    writer.add_figure("pairwise/dist_ot_panel", fig, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig, out_dir=save_dir, tag="pairwise__dist_ot_panel", step=step)
    plt.close(fig)


def tb_information_matrix_figs(
    writer,
    step: int,
    *,
    stat_tracker,
    traversal_sets,
    recognizer,
    z_bd: torch.Tensor,
    dt: torch.Tensor,
    save_dir: str | Path | None = None,
):
    """
    Compute + log a 2x2 information/confusion panel:
      - Λ heatmap (squared dot products between per-label step directions)
      - p(pred|true) confusion heatmap
      - Q(Λ) history
      - d_eff(Σ) history
    """
    lam, conf = ImageViz.compute_lambda_and_confusion(
        traversal_sets=traversal_sets,
        recognizer=recognizer,
        z_bd=z_bd,
        dt=dt,
    )
    stat_tracker.update_information_metrics(step_idx=int(step), lambda_kxk=lam, conf_prob_kxk=conf)

    fig = ImageViz.plot_information_matrix_panel(
        lambda_kxk=lam,
        conf_prob_kxk=conf,
        steps=stat_tracker.info_steps,
        Q_lambda_hist=stat_tracker.info_Q_lambda_hist,
        Q_conf_hist=stat_tracker.info_Q_conf_hist,
        deff_lambda_hist=stat_tracker.info_deff_lambda_hist,
        deff_conf_hist=stat_tracker.info_deff_conf_hist,
        title="Information/Confusion panel (batch-based) with disentanglement invariants",
    )
    writer.add_figure("info/info_matrix_panel", fig, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig, out_dir=save_dir, tag="info__info_matrix_panel", step=step)
    plt.close(fig)


class PathImages(data.Dataset):
    def __init__(self, root_path):
        self.images_files = glob.glob(osp.join(root_path, '*.jpg'))
        self.images_files.sort()

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        return self.image2tensor(self.images_files[index])

    @staticmethod
    def image2tensor(image_file):
        # Open image in BGR order and convert to RBG order
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')
        return torch.tensor(np.transpose(img, (2, 0, 1))).float()

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from losses.diffaug import DiffAugment as aug
from cat_pyramid import (
    build_cat_fake_pyramid,
    build_cat_real_pyramid,
    cat_consistency_loss,
)


CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_generator_attr(generator, name):
    generator_module = getattr(generator, "module", generator)
    return getattr(generator_module, name)


def scale_wise_relativistic_g_loss(real_logits, fake_logits):
    losses = []
    for k in range(real_logits.shape[1]):
        rel = fake_logits[:, k] - real_logits[:, k]
        losses.append(F.softplus(-rel))
    return torch.stack(losses, dim=1).mean(dim=1)


def scale_wise_relativistic_d_loss(real_logits, fake_logits):
    losses = []
    for k in range(real_logits.shape[1]):
        rel = real_logits[:, k] - fake_logits[:, k]
        losses.append(F.softplus(-rel))
    return torch.stack(losses, dim=1).mean(dim=1)


def preprocess_raw_image(x, enc_type):
    if "clip" in enc_type:
        x = x / 255.0
        x = F.interpolate(x, 224, mode="bicubic")
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif "mocov3" in enc_type or "mae" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif "dinov2" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = F.interpolate(x, 224, mode="bicubic")
    elif "dinov1" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif "jepa" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = F.interpolate(x, 224, mode="bicubic")
    return x


def resize_spatial(tensor: torch.Tensor, H_out: int, W_out: int = None):
    B, L, C = tensor.shape
    H_in = W_in = int(np.sqrt(L))
    assert H_in * W_in == L, f"L={L} cannot be reshaped to square"
    if W_out is None:
        W_out = H_out
    x = tensor.transpose(1, 2).reshape(B, C, H_in, W_in)
    x = F.interpolate(x, size=(H_out, W_out), mode="bicubic", align_corners=False)
    return x.reshape(B, C, -1).transpose(1, 2)


class CATLoss:
    def __init__(
        self,
        encoders=None,
        encoder_types=None,
        architectures=None,
        accelerator=None,
        r1_gamma=1.0,
        r1_every=1,
        r2_gamma=1.0,
        r2_every=1,
        lambda_repa=1.0,
        lambda_cons=0.1,
        cons_weights=(1 / 3, 1 / 2, 1.0),
        gp_eps=0.01,
        gp_batch_frac=0.25,
    ):
        self.encoders = encoders or []
        self.encoder_types = encoder_types or []
        self.architectures = architectures or []
        self.accelerator = accelerator
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        self.r1_every = r1_every
        self.r2_every = r2_every
        self.lambda_repa = lambda_repa
        self.lambda_cons = lambda_cons
        self.cons_weights = cons_weights
        self.gp_eps = gp_eps
        self.gp_batch_frac = gp_batch_frac
        self.policy = "color,translation,cutout,flip"
        self.policy_raw_image = "translation,flip"
        self.aug_prob = 1.0

    def _sample_z(self, generator, batch_size, device, dtype):
        return torch.randn(
            batch_size,
            get_generator_attr(generator, "latent_size"),
            device=device,
            dtype=dtype,
        )

    def _ensure_model_kwargs(self, generator, images, model_kwargs):
        if model_kwargs is None:
            model_kwargs = {}
        if "z" not in model_kwargs:
            model_kwargs["z"] = self._sample_z(generator, images.shape[0], images.device, images.dtype)
        if "x" not in model_kwargs:
            model_kwargs["x"] = torch.randn_like(images)
        return model_kwargs

    def _augment_pyramid(self, pyramid):
        pyramid_aug = []
        aug_params = None
        for scale in pyramid:
            scale_aug, params = aug(scale, prob=self.aug_prob, policy=self.policy)
            pyramid_aug.append(scale_aug)
            if aug_params is None:
                aug_params = params
        return pyramid_aug, aug_params

    def _gp_indices(self, batch_size, device):
        gp_bs = max(1, int(batch_size * self.gp_batch_frac))
        return torch.randperm(batch_size, device=device)[:gp_bs]

    def _approx_gp_pyramid(self, discriminator, pyramid, logits, y, idx):
        perturbed = [
            scale[idx] + self.gp_eps * torch.randn_like(scale[idx])
            for scale in pyramid
        ]
        pert_logits = discriminator(perturbed, y=y[idx])
        base_score = logits[idx].sum(dim=1)
        pert_score = pert_logits.sum(dim=1)
        return ((pert_score - base_score) / self.gp_eps).pow(2).mean()

    def encode_feature(self, raw_image, do_aug=True, aug_params=None):
        zs = []
        with self.accelerator.autocast():
            with torch.no_grad():
                for encoder, encoder_type, _arch in zip(
                    self.encoders, self.encoder_types, self.architectures
                ):
                    raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                    if do_aug:
                        raw_image_, _ = aug(
                            raw_image_, aug_params=aug_params, policy=self.policy_raw_image
                        )
                    z = encoder.forward_features(raw_image_)
                    assert "dinov2" in encoder_type
                    zs.append(z)
        return zs

    def _repa_loss(self, real_aux, raw_images, aug_params, device):
        if self.lambda_repa <= 0 or len(self.encoders) == 0 or real_aux["x_feat"] is None:
            return torch.zeros((), device=device)

        z_feat_tilde = real_aux["x_feat"]
        z_feat = self.encode_feature(raw_images, do_aug=True, aug_params=aug_params)[0]

        z_cls_kd_loss = 1.0 - F.cosine_similarity(
            z_feat_tilde[0].squeeze(1), z_feat["x_norm_clstoken"].detach(), dim=-1
        )
        z_feat_spatial = z_feat["x_norm_patchtokens"]
        if z_feat_tilde[1].shape != z_feat_spatial.shape:
            z_feat_spatial = resize_spatial(
                z_feat_spatial, H_out=int(np.sqrt(z_feat_tilde[1].shape[1]))
            )
        z_spatial_kd_loss = (
            1.0 - F.cosine_similarity(z_feat_tilde[1], z_feat_spatial.detach(), dim=-1)
        ).sum(dim=1) / z_feat_spatial.shape[1]
        return z_cls_kd_loss + z_spatial_kd_loss

    def step_gen(
        self,
        generator,
        discriminator,
        discriminator_ema,
        images,
        raw_images,
        global_step,
        model_kwargs=None,
        zs=None,
        **kwargs,
    ):
        model_kwargs = self._ensure_model_kwargs(generator, images, model_kwargs)

        stage_outputs = generator(update_ema=True, return_stages=True, **model_kwargs)
        fake_pyramid = build_cat_fake_pyramid(stage_outputs)
        fake_pyramid, _ = self._augment_pyramid(fake_pyramid)

        real_pyramid = build_cat_real_pyramid(images.detach())
        real_pyramid, _ = self._augment_pyramid(real_pyramid)

        with torch.no_grad():
            real_logits = discriminator(real_pyramid, y=model_kwargs["y"])

        fake_logits = discriminator(fake_pyramid, y=model_kwargs["y"])

        g_adv = scale_wise_relativistic_g_loss(real_logits, fake_logits)
        cons_loss = cat_consistency_loss(stage_outputs, weights=self.cons_weights)
        loss = g_adv + self.lambda_cons * cons_loss

        loss_dict = {
            "gen_loss": loss,
            "g_adv": g_adv,
            "cons_loss": cons_loss,
        }
        extras = {"gen_images": stage_outputs}
        return loss, loss_dict, extras

    def step_disc(
        self,
        generator,
        discriminator,
        discriminator_ema,
        images,
        raw_images,
        global_step,
        model_kwargs=None,
        zs=None,
        **kwargs,
    ):
        model_kwargs = self._ensure_model_kwargs(generator, images, model_kwargs)

        with torch.no_grad():
            stage_outputs = generator(update_ema=False, return_stages=True, **model_kwargs)
            fake_pyramid = build_cat_fake_pyramid(stage_outputs)

        real_pyramid = build_cat_real_pyramid(images)
        fake_pyramid, gen_aug_params = self._augment_pyramid(fake_pyramid)
        real_pyramid, real_aug_params = self._augment_pyramid(real_pyramid)

        real_logits, real_aux = discriminator(real_pyramid, y=model_kwargs["y"], return_aux=True)
        fake_logits = discriminator(fake_pyramid, y=model_kwargs["y"])

        d_adv = scale_wise_relativistic_d_loss(real_logits, fake_logits)
        repa_loss = self._repa_loss(real_aux, raw_images, real_aug_params, images.device)

        loss = d_adv + self.lambda_repa * repa_loss
        loss_dict = {
            "disc_loss": d_adv,
            "d_adv": d_adv,
            "repa_loss": repa_loss,
        }

        batch_size = images.shape[0]
        gp_idx = self._gp_indices(batch_size, images.device)

        if global_step % self.r1_every == 0:
            r1_loss = self._approx_gp_pyramid(
                discriminator, real_pyramid, real_logits, model_kwargs["y"], gp_idx
            )
            loss = loss + self.r1_gamma / 2 * r1_loss
            loss_dict["r1_loss"] = r1_loss

        if global_step % self.r2_every == 0:
            r2_loss = self._approx_gp_pyramid(
                discriminator, fake_pyramid, fake_logits, model_kwargs["y"], gp_idx
            )
            loss = loss + self.r2_gamma / 2 * r2_loss
            loss_dict["r2_loss"] = r2_loss

        return loss, loss_dict, {}

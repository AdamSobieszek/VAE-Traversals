#!/usr/bin/env python3
"""Smoke tests for the CAT implementation."""

import sys
import traceback

import torch

sys.path.insert(0, ".")

from cat_pyramid import (
    build_block_diag_attention_mask,
    build_cat_fake_pyramid,
    build_cat_real_pyramid,
    cat_consistency_loss,
)
from losses import CATLoss
from models.discriminator import CATD_models
from models.generator import CAT_models

BLOCK_KWARGS = {"fused_attn": True, "qk_norm": True}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_batch(device, batch_size=2):
    images = torch.randn(batch_size, 4, 32, 32, device=device)
    raw_images = torch.randint(0, 255, (batch_size, 3, 256, 256), device=device, dtype=torch.uint8)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    return images, raw_images, y


def test_registry():
    assert set(CAT_models.keys()) == {"CAT-G-B/2", "CAT-G-M/2", "CAT-G-H/2"}
    assert set(CATD_models.keys()) == {"CAT-D-B/2"}
    print("registry: ok")


def test_generator_variants(device, block_kwargs):
    B = 2
    configs = {
        "CAT-G-B/2": (3, 6, 9, 12),
        "CAT-G-M/2": (6, 12, 18, 24),
        "CAT-G-H/2": (8, 16, 24, 32),
    }
    for name, expected_layers in configs.items():
        G = CAT_models[name](input_size=32, num_classes=1000, z_dims=[768], **block_kwargs).to(device)
        assert tuple(G.output_layers) == expected_layers

        x = torch.zeros(B, 4, 32, 32, device=device)
        y = torch.randint(0, 1000, (B,), device=device)
        z = torch.randn(B, G.latent_size, device=device)

        stages = G(x=x, y=y, z=z, return_stages=True)
        assert len(stages) == 4
        for h in stages:
            assert h.shape == (B, 4, 32, 32)

        final = G(x=x, y=y, z=z, return_stages=False)
        assert final.shape == (B, 4, 32, 32)
        assert torch.allclose(final, stages[-1], atol=1e-5, rtol=1e-4)
        print(f"generator {name}: ok")


def test_pyramid_and_discriminator(device, block_kwargs):
    B = 2
    G = CAT_models["CAT-G-B/2"](input_size=32, num_classes=1000, z_dims=[768], **block_kwargs).to(device)
    D = CATD_models["CAT-D-B/2"](num_classes=1000, z_dims=[768], **block_kwargs).to(device)

    x = torch.zeros(B, 4, 32, 32, device=device)
    y = torch.randint(0, 1000, (B,), device=device)
    z = torch.randn(B, G.latent_size, device=device)

    stages = G(x=x, y=y, z=z, return_stages=True)
    fake_pyramid = build_cat_fake_pyramid(stages)
    assert [t.shape[-1] for t in fake_pyramid] == [32, 16, 8, 4]

    real_latent = torch.randn(B, 4, 32, 32, device=device)
    real_pyramid = build_cat_real_pyramid(real_latent)
    assert [t.shape[-1] for t in real_pyramid] == [32, 16, 8, 4]

    fake_logits = D(fake_pyramid, y)
    assert fake_logits.shape == (B, 4)

    real_logits, aux = D(real_pyramid, y, return_aux=True)
    assert real_logits.shape == (B, 4)
    assert aux["x_feat"] is not None
    assert aux["x_feat"][0].shape[0] == B
    assert aux["x_feat"][1].shape[0] == B
    print("pyramid + discriminator: ok")


def test_attention_mask(device):
    group_lengths = [257, 65, 17, 5]
    mask = build_block_diag_attention_mask(group_lengths, device, torch.float32)
    assert mask.shape == (344, 344)
    assert mask[0, 0].item() == 0
    assert mask[257, 257].item() == 0
    assert mask[0, 300].item() == float("-inf")
    print("attention mask: ok")


def test_consistency_loss(device):
    B = 2
    stages = [torch.randn(B, 4, 32, 32, device=device, requires_grad=True) for _ in range(4)]
    loss = cat_consistency_loss(stages)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    for stage in stages:
        assert stage.grad is not None
        assert torch.isfinite(stage.grad).all()
    print("consistency loss + backward: ok")


def test_loss_steps(device, block_kwargs):
    B = 2
    G = CAT_models["CAT-G-B/2"](input_size=32, num_classes=1000, z_dims=[0], **block_kwargs).to(device)
    D = CATD_models["CAT-D-B/2"](num_classes=1000, z_dims=[0], **block_kwargs).to(device)
    loss_fn = CATLoss(encoders=[], encoder_types=[], architectures=[], lambda_repa=0.0)

    images, raw_images, y = make_batch(device, B)
    model_kwargs = {"y": y}

    d_loss, d_dict, _ = loss_fn.step_disc(G, D, None, images, raw_images, 0, model_kwargs)
    assert torch.isfinite(d_loss).all()
    for key in ("disc_loss", "d_adv", "r1_loss", "r2_loss"):
        assert key in d_dict
        assert torch.isfinite(d_dict[key]).all()

    g_loss, g_dict, extras = loss_fn.step_gen(G, D, None, images, raw_images, 0, model_kwargs)
    assert torch.isfinite(g_loss).all()
    assert torch.isfinite(g_dict["g_adv"]).all()
    assert torch.isfinite(g_dict["cons_loss"]).all()
    assert len(extras["gen_images"]) == 4
    print("loss steps: ok")


def test_backward_paths(device, block_kwargs):
    B = 2
    G = CAT_models["CAT-G-B/2"](input_size=32, num_classes=1000, z_dims=[0], **block_kwargs).to(device)
    D = CATD_models["CAT-D-B/2"](num_classes=1000, z_dims=[0], **block_kwargs).to(device)
    loss_fn = CATLoss(encoders=[], encoder_types=[], architectures=[], lambda_repa=0.0)

    images, raw_images, y = make_batch(device, B)
    model_kwargs = {"y": y}

    for p in G.parameters():
        p.grad = None
    for p in D.parameters():
        p.grad = None

    D.requires_grad_(True)
    d_loss, _, _ = loss_fn.step_disc(G, D, None, images, raw_images, 0, model_kwargs)
    d_loss.mean().backward()
    d_grad = sum(p.grad.abs().sum().item() for p in D.parameters() if p.grad is not None)
    g_grad_after_d = sum(p.grad.abs().sum().item() for p in G.parameters() if p.grad is not None)
    assert d_grad > 0
    assert g_grad_after_d == 0

    for p in G.parameters():
        p.grad = None
    for p in D.parameters():
        p.grad = None

    D.requires_grad_(False)
    G.requires_grad_(True)
    g_loss, _, _ = loss_fn.step_gen(G, D, None, images, raw_images, 0, model_kwargs)
    g_loss.mean().backward()
    g_grad = sum(p.grad.abs().sum().item() for p in G.parameters() if p.grad is not None)
    d_grad_after_g = sum(p.grad.abs().sum().item() for p in D.parameters() if p.grad is not None)
    assert g_grad > 0
    assert d_grad_after_g == 0
    print("backward paths: ok")


def test_all_generators_with_base_discriminator(device, block_kwargs):
    B = 2
    D = CATD_models["CAT-D-B/2"](num_classes=1000, z_dims=[0], **block_kwargs).to(device)
    images, _, y = make_batch(device, B)

    for name in CAT_models:
        G = CAT_models[name](input_size=32, num_classes=1000, z_dims=[0], **block_kwargs).to(device)
        z = torch.randn(B, G.latent_size, device=device)
        x = torch.randn_like(images)
        stages = G(x=x, y=y, z=z, return_stages=True)
        fake_pyramid = build_cat_fake_pyramid(stages)
        logits = D(fake_pyramid, y)
        assert logits.shape == (B, 4)
        print(f"pair {name} + CAT-D-B/2: ok")


def test_train_imports():
    import train
    import generate

    args = train.parse_args(
        [
            "--exp-name",
            "smoke",
            "--model",
            "CAT-G-B/2",
            "--modelD",
            "CAT-D-B/2",
        ]
    )
    assert args.model == "CAT-G-B/2"
    assert args.modelD == "CAT-D-B/2"
    assert args.learning_rate == 2e-4
    assert args.batch_size == 512
    assert "CAT_models" in generate.__dict__ or hasattr(generate, "main")
    print("train/generate imports: ok")


def main():
    device = pick_device()
    block_kwargs = BLOCK_KWARGS
    print(f"Running CAT smoke tests on device={device}, block_kwargs={block_kwargs}")

    tests = [
        test_registry,
        lambda: test_generator_variants(device, block_kwargs),
        lambda: test_pyramid_and_discriminator(device, block_kwargs),
        lambda: test_attention_mask(device),
        lambda: test_consistency_loss(device),
        lambda: test_loss_steps(device, block_kwargs),
        lambda: test_backward_paths(device, block_kwargs),
        lambda: test_all_generators_with_base_discriminator(device, block_kwargs),
        test_train_imports,
    ]

    for test in tests:
        try:
            test()
        except Exception:
            print(f"FAILED: {test.__name__ if hasattr(test, '__name__') else test}")
            traceback.print_exc()
            sys.exit(1)

    print(f"All CAT smoke tests passed on {device}.")


if __name__ == "__main__":
    main()

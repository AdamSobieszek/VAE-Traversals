"""Regression tests for PCA-residual StyleGAN feature codec."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.stylegan_feature_codec import (
    FEATURE_CHANNELS,
    LATENT_CHANNELS,
    FrozenRank4PCA,
    PCAResidualFeatureCodec,
    StreamingPCACalibrator,
    feature_codec_loss,
)


def _synthetic_features(batch=2, h=48, w=48, noise=0.02):
    c, k = FEATURE_CHANNELS, LATENT_CHANNELS
    basis, _ = torch.linalg.qr(torch.randn(c, k))
    codes = torch.randn(batch, k, h, w)
    x = torch.einsum("ck,bkhw->bchw", basis, codes) + noise * torch.randn(batch, c, h, w)
    return x, basis


def test_pca_roundtrip_and_baseline():
    x, _ = _synthetic_features()
    calib = StreamingPCACalibrator(FEATURE_CHANNELS)
    for _ in range(12):
        xx, _ = _synthetic_features()
        calib.update(xx, max_samples=4096)
    pca = calib.finalize(LATENT_CHANNELS)
    assert bool(pca.fitted.item())
    recon = pca.reconstruct(x)
    mse = float(F.mse_loss(recon, x))
    assert mse < 0.05, mse
    z = pca.encode(x)
    assert z.shape[1] == LATENT_CHANNELS
    fast_mse = pca.reconstruction_mse(x, z_pca=z)
    assert torch.allclose(fast_mse, F.mse_loss(recon, x), rtol=1e-4, atol=1e-7)
    # encode/decode identity on PCA subspace approximately
    recon2 = pca.decode(z)
    assert float(F.mse_loss(recon, recon2)) < 1e-6


def test_zero_gate_equals_pca():
    x, _ = _synthetic_features()
    calib = StreamingPCACalibrator(FEATURE_CHANNELS)
    for _ in range(10):
        xx, _ = _synthetic_features()
        calib.update(xx, max_samples=2048)
    pca = calib.finalize(LATENT_CHANNELS)
    codec = PCAResidualFeatureCodec(
        pca, hidden_channels=32, num_blocks=2, use_context=False, drop_path=0.0
    )
    out = codec(x, sample_posterior=False, deterministic=True)
    diff = float(F.mse_loss(out["recon"], pca.reconstruct(x)))
    assert diff < 1e-6, diff


def test_shapes_and_train_step():
    x, _ = _synthetic_features(batch=2, h=32, w=32)
    calib = StreamingPCACalibrator(FEATURE_CHANNELS)
    for _ in range(8):
        xx, _ = _synthetic_features(h=32, w=32)
        calib.update(xx, max_samples=2048)
    pca = calib.finalize(LATENT_CHANNELS)
    codec = PCAResidualFeatureCodec(
        pca, hidden_channels=32, num_blocks=2, use_context=True, drop_path=0.0
    )
    out = codec(x, sample_posterior=True, deterministic=False)
    assert out["recon"].shape == x.shape
    assert out["z"].shape == (2, LATENT_CHANNELS, 32, 32)
    loss, stats = feature_codec_loss(
        out["recon"], x, out["posterior"], pca,
        z=out["z"], z_pca=out["z_pca"], mean_res=out["mean_res"],
        kl_weight=1e-3, deterministic=False,
    )
    assert torch.isfinite(loss)
    assert "nmse" in stats
    loss.backward()
    assert codec.decoder.to_out.weight.grad is not None


def test_compact_pca_serialization():
    x, _ = _synthetic_features()
    calib = StreamingPCACalibrator(FEATURE_CHANNELS)
    for _ in range(6):
        xx, _ = _synthetic_features()
        calib.update(xx, max_samples=1024)
    pca = calib.finalize(LATENT_CHANNELS)
    state = pca.state_dict_compact()
    pca2 = FrozenRank4PCA(FEATURE_CHANNELS, LATENT_CHANNELS)
    pca2.load_compact(state)
    a = pca.reconstruct(x)
    b = pca2.reconstruct(x)
    assert float(F.mse_loss(a, b)) < 1e-8


if __name__ == "__main__":
    test_pca_roundtrip_and_baseline()
    test_zero_gate_equals_pca()
    test_shapes_and_train_step()
    test_compact_pca_serialization()
    print("All stylegan_feature_codec tests passed.")

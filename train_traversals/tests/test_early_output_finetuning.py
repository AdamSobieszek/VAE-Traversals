"""Bounded correctness checks; image/decoder checks use MPS at batch one locally."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import StyleGAN_SkipNetwork128 as training
from models.StyleGAN2_mps.early_output_model import (
    PointwiseStyleDecoder, load_generator_checkpoint, build_optimized_early_output_synthesis,
)
from models.StyleGAN2_mps.torch_utils.ops.inference_opt import InferenceOptConfig


class FinetuningTests(unittest.TestCase):
    def test_scaler_skips_reference_pull_on_overflow(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = training.AdamWWithReferenceDecay(
            [dict(params=[parameter], weight_decay=0., reference_weight_decay=5.)],
            reference_parameters={id(parameter): torch.ones(1)}, lr=.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.)
        scaler.scale((parameter * float("inf")).sum()).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(parameter.item(), 2.)
        self.assertEqual(scaler.get_scale(), 4.)

    def test_reference_decay_and_missing_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1.1, 0.9]))
        unused = torch.nn.Parameter(torch.tensor([2.0]))
        reference = torch.ones(2)
        initial = p.detach().clone()
        optimizer = training.AdamWWithReferenceDecay(
            [dict(params=[p, unused], weight_decay=0., reference_weight_decay=2.)],
            reference_parameters={id(p): reference, id(unused): torch.zeros(1)},
            lr=0.01, betas=(0., 0.), eps=1e-8)
        p.grad = torch.tensor([0.5, -0.5])
        optimizer.step()
        expected = initial.lerp(reference, .02) - .01 * p.grad / (p.grad.abs() + 1e-8)
        torch.testing.assert_close(p, expected)
        self.assertEqual(unused.item(), 2.)
        torch.testing.assert_close(reference, torch.ones(2))

    def test_weighted_reduction_precision_and_sample_normalization(self):
        values = torch.full((2, 1, 4, 4), 1e-4, dtype=torch.float16, requires_grad=True)
        mask = torch.full((2, 1, 4, 4), 1e-5)
        mask[1] *= 2
        result = training.weighted_spatial_mean(values, mask)
        torch.testing.assert_close(result, values.float().mean())
        result.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        mask = training.face_saliency_mask(128, 128, inner_mass=.5)
        self.assertFalse(torch.equal(mask, training.face_saliency_mask(128, 128, inner_mass=.8)))

    def test_private_balanced_sampler(self):
        class FakeMapping:
            w_avg = torch.zeros(512)
            def __call__(self, z, *args, **kwargs):
                return z[:, None].repeat(1, 18, 1)
        extractor = SimpleNamespace(G=SimpleNamespace(mapping=FakeMapping()))
        extractor.map_z = lambda z: extractor.G.mapping(z)
        cfg = training.TrainConfig(batch_size=24)
        a = torch.Generator().manual_seed(123)
        b = torch.Generator().manual_seed(123)
        ws1, labels1 = training._sample_training_ws(extractor, cfg, a, torch.device("cpu"))
        torch.randn(100)
        ws2, labels2 = training._sample_training_ws(extractor, cfg, b, torch.device("cpu"))
        torch.testing.assert_close(ws1, ws2)
        self.assertEqual(torch.bincount(labels1).tolist(), [12, 4, 4, 4])
        self.assertTrue(torch.equal(labels1, labels2))
        for i in range(11, 18):
            torch.testing.assert_close(ws1[:, i], ws1[:, 9 if i % 2 else 10])

    @unittest.skipUnless(torch.backends.mps.is_available(), "Local decoder tests require MPS")
    def test_style_and_feature_gradients_and_reload_mps(self):
        torch.manual_seed(7)
        decoder = PointwiseStyleDecoder().to("mps")
        features = torch.randn(1, 256, 16, 16, device="mps", requires_grad=True)
        ws = torch.randn(1, 12, 512, device="mps", requires_grad=True)
        decoder(features, ws).square().mean().backward()
        for gradient in (features.grad, ws.grad[:, 9:11], decoder.style[1].weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        restored = PointwiseStyleDecoder().to("mps")
        restored.load_state_dict(decoder.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(features, ws), decoder(features, ws), rtol=0, atol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Local inference tests require MPS")
    def test_optimized_output_and_latent_gradient_mps(self):
        if not Path(training.DEFAULT_EARLY_OUTPUT_WEIGHTS).is_file():
            self.skipTest("Trained checkpoint is not installed")
        model = load_generator_checkpoint(training.DEFAULT_EARLY_OUTPUT_WEIGHTS, "mps")
        torch.manual_seed(13)
        with torch.no_grad():
            ws = model.mapping(torch.randn(1, 512, device="mps"), None)
        ws = ws.clone().requires_grad_(True)
        expected = model.synthesis(ws, noise_mode="const", fused_modconv=False)
        gradient = torch.autograd.grad(expected.square().mean(), ws)[0]
        decoder = model.synthesis.decoder
        optimized = build_optimized_early_output_synthesis(
            model.synthesis, InferenceOptConfig(shared_modconv=True, fir_compose=False,
                                               fused_modconv=False), copy_module=False)
        self.assertIs(model.synthesis.decoder, decoder)
        actual = optimized(ws, noise_mode="const", force_fp32=True)
        actual_gradient = torch.autograd.grad(actual.square().mean(), ws)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(actual_gradient, gradient, rtol=2e-3, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

"""Bounded correctness checks; image/decoder checks use MPS at batch one locally."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import StyleGAN_SkipNetwork128 as training
from models.StyleGAN2_mps.early_output_model import FusedFastEarlyOutputDecoder


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
    def test_migration_gradients_and_decoder_reload_mps(self):
        device = torch.device("mps")
        torch.manual_seed(7)
        old = FusedFastEarlyOutputDecoder(256, 16).to(device)
        with torch.no_grad():
            old.linear.weight.normal_(std=.01)
            old.mix_out.weight.normal_(std=.01)
        features = torch.randn(1, 256, 16, 16, device=device)
        ws = torch.randn(1, 12, 512, device=device)
        with torch.no_grad():
            expected = old(features, ws)
        for architecture, hidden in (("pointwise", 16), ("pointwise", 32), ("pointwise_style", 16), ("spatial_style", 16)):
            model = SimpleNamespace(synthesis=SimpleNamespace(decoder=copy.deepcopy(old)))
            cfg = training.TrainConfig(hidden_channels=hidden, decoder_architecture=architecture)
            training.migrate_finetuning_decoder(model, cfg)
            decoder = model.synthesis.decoder
            if architecture == "pointwise_style":
                self.assertIsInstance(decoder.spatial_conv, torch.nn.Identity)
            with torch.no_grad():
                torch.testing.assert_close(decoder(features, ws), expected, rtol=2e-5, atol=2e-6)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            target = torch.randn_like(expected)
            for _ in range(4):
                optimizer.zero_grad()
                (decoder(features, ws) - target).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None))
                optimizer.step()
            if architecture != "pointwise":
                self.assertGreater(float(decoder.spatial_in.weight.grad.abs().sum()), 0)
                self.assertGreater(float(decoder.style[1].weight.grad.abs().sum()), 0)
                with self.assertRaisesRegex(ValueError, "W slots"):
                    decoder(features)
                with torch.no_grad():
                    self.assertFalse(torch.equal(decoder(features, ws), decoder(features, -ws)))
            restored = FusedFastEarlyOutputDecoder(256, hidden, architecture=architecture).to(device)
            restored.load_state_dict(decoder.state_dict(), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(features, ws), decoder(features, ws), rtol=0, atol=0)
                # Continuing a trained conditioned head must retain its branch.
                expected_trained = decoder(features, ws)
                training.migrate_finetuning_decoder(model, cfg)
                torch.testing.assert_close(model.synthesis.decoder(features, ws), expected_trained, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

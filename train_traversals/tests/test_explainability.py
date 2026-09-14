"""Mathematical checks for the post-hoc joint-AGOP estimators."""

import numpy as np
import torch

from lib.explainability import _gradients, covariance_modes


def test_exact_and_matrix_free_spectra():
    rows = np.random.default_rng(2).normal(size=(40, 12)).astype('float32')
    expected = np.linalg.eigvalsh(rows.T @ rows / len(rows))[::-1][:3]
    for exact_max, iterations, tolerance in ((100, 1, 2e-5), (10, 30, 2e-3)):
        modes, values, residuals, _, _ = covariance_modes(
            rows, (1, 2, 3), 3, exact_max, iterations)
        np.testing.assert_allclose(values, expected, rtol=tolerance)
        assert modes.shape == (3, 2, 1, 2, 3)
        assert residuals.max() < .02


def test_orthogonal_rows_do_not_vanish_after_threshold():
    _, values, _, _, method = covariance_modes(
        np.eye(10, dtype='float32'), (1, 1, 5), 1, 8, 2)
    np.testing.assert_allclose(values, [.1], rtol=1e-5)
    assert method.startswith('matrix-free')


def test_diagonal_pair_keeps_independent_endpoint_gradients():
    class Trainer:
        @staticmethod
        def _pair_logits(_, first, second):
            difference = (second - first).flatten(1)
            return torch.stack((difference[:, 0], difference[:, 1]), dim=1)

    image = torch.randn(3, 1, 2, 2)
    score, (first, second, common, relative) = _gradients(
        Trainer(), None, 0, 1, image, image)
    torch.testing.assert_close(score, torch.zeros_like(score))
    torch.testing.assert_close(common, torch.zeros_like(common))
    torch.testing.assert_close(first, -second)
    assert torch.count_nonzero(relative)

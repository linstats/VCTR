from __future__ import annotations

import unittest

import numpy as np

from src.data import PairedEyeDataset
from src.dgps import PairedCase1DGP


class PairedCase1DGPTests(unittest.TestCase):
    def test_sample_returns_paired_dataset_with_expected_shapes(self) -> None:
        dataset = PairedCase1DGP(
            n_subject=5,
            R=3,
            S=4,
            p0=2,
            sigma2=2.0,
            rho=0.25,
        ).sample(seed=123)

        self.assertIsInstance(dataset, PairedEyeDataset)
        self.assertEqual(dataset.X.shape, (5, 2, 3, 4))
        self.assertEqual(dataset.y.shape, (5, 2))
        self.assertEqual(dataset.Z.shape, (5, 2))
        self.assertEqual(dataset.t.shape, (5,))
        self.assertEqual(dataset.A_true.shape, (5, 3, 4))
        self.assertEqual(dataset.beta_true.shape, (2,))
        self.assertEqual(dataset.Sigma_true.shape, (2, 2))

    def test_sigma_true_matches_exchangeable_structure(self) -> None:
        sigma2 = 1.7
        rho = 0.4
        dataset = PairedCase1DGP(n_subject=3, sigma2=sigma2, rho=rho).sample(seed=0)
        np.testing.assert_allclose(
            dataset.Sigma_true,
            sigma2 * np.array([[1.0, rho], [rho, 1.0]]),
        )

    def test_to_iid_observations_is_compatible(self) -> None:
        dataset = PairedCase1DGP(n_subject=4, R=2, S=3, p0=2).sample(seed=7)
        flat = dataset.to_iid_observations()
        self.assertEqual(flat.X.shape, (8, 2, 3))
        self.assertEqual(flat.y.shape, (8,))
        self.assertEqual(flat.t.shape, (8,))
        self.assertEqual(flat.Z.shape, (8, 2))

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            PairedCase1DGP(coef_type="invalid").sample(seed=0)
        with self.assertRaises(ValueError):
            PairedCase1DGP(sigma2=0.0).sample(seed=0)
        with self.assertRaises(ValueError):
            PairedCase1DGP(rho=1.0).sample(seed=0)
        with self.assertRaises(ValueError):
            PairedCase1DGP(eye_ids=(0, 1, 2)).sample(seed=0)
        with self.assertRaises(ValueError):
            PairedCase1DGP(p0=2, beta_true=(3.0,)).sample(seed=0)

    def test_default_beta_true_matches_case1_paper_setting(self) -> None:
        dataset = PairedCase1DGP(n_subject=3, p0=2).sample(seed=0)
        np.testing.assert_allclose(dataset.beta_true, np.array([3.0, 3.0]))

    def test_explicit_beta_true_is_respected(self) -> None:
        dataset = PairedCase1DGP(n_subject=3, p0=2, beta_true=(3.0, 1.5)).sample(seed=0)
        np.testing.assert_allclose(dataset.beta_true, np.array([3.0, 1.5]))

    def test_sampling_is_reproducible_for_fixed_seed(self) -> None:
        dgp = PairedCase1DGP(n_subject=6, R=2, S=2, p0=1, sigma2=1.2, rho=0.1)
        d1 = dgp.sample(seed=2026)
        d2 = dgp.sample(seed=2026)
        np.testing.assert_allclose(d1.t, d2.t)
        np.testing.assert_allclose(d1.Z, d2.Z)
        np.testing.assert_allclose(d1.X, d2.X)
        np.testing.assert_allclose(d1.y, d2.y)
        np.testing.assert_allclose(d1.A_true, d2.A_true)
        np.testing.assert_allclose(d1.beta_true, d2.beta_true)
        np.testing.assert_allclose(d1.Sigma_true, d2.Sigma_true)


if __name__ == "__main__":
    unittest.main()

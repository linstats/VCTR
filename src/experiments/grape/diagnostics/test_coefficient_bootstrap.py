"""Focused tests for fixed-grid estimation and GRAPE bootstrap sampling."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import PairedEyeDataset
from src.experiments.grape.diagnostics.bootstrap_coefficients import (
    fit_coefficients_on_grid,
    pair_row_resample,
    patient_cluster_resample,
    select_z_columns,
)
from src.experiments.grape.diagnostics.compare_xz_beta_bootstrap import significance_label
from src.models import PairedEyeVCTRModel
from src.models.covariance import smooth_variance_curve


class FixedGridStage3Test(unittest.TestCase):
    def test_public_fixed_grid_matches_full_refit_targets(self) -> None:
        rng = np.random.default_rng(42)
        n_pair = 24
        t = np.linspace(0.0, 1.0, num=n_pair)
        X = rng.normal(size=(n_pair, 2, 1, 2))
        true_A = np.column_stack([0.7 + 0.3 * t, -0.4 + 0.2 * t])
        y = np.sum(X.reshape(n_pair, 2, 2) * true_A[:, None, :], axis=2)
        y += rng.normal(scale=0.15, size=y.shape)
        dataset = PairedEyeDataset(
            subject_ids=np.asarray([f"pair_{idx}" for idx in range(n_pair)]),
            eye_ids=np.asarray(["OD", "OS"]),
            t=t,
            X=X,
            Z=np.empty((n_pair, 0), dtype=float),
            y=y,
        )
        model = PairedEyeVCTRModel(
            covariance_mode="exchangeable_varying_sigma",
            a_eval_mode="full",
            signal_bandwidth=0.8,
            variance_bandwidth=0.8,
            ridge=1e-5,
        )
        initial = model.initial_fit_iid(dataset)
        covariance = model.estimate_covariance(dataset, initial)
        direct_A, beta_local = model.estimate_stage3_A_at(dataset, covariance, initial, dataset.t)
        fitted = model.refit_with_covariance(dataset, covariance, initial)

        self.assertEqual(direct_A.shape, (n_pair, 1, 2))
        self.assertEqual(beta_local.shape, (n_pair, 0))
        np.testing.assert_allclose(direct_A, fitted.A_hat, rtol=1e-11, atol=1e-11)

    def test_xz_fit_returns_fixed_grid_A_and_final_beta(self) -> None:
        rng = np.random.default_rng(123)
        n_pair = 40
        t = np.linspace(0.0, 1.0, num=n_pair)
        X = rng.normal(size=(n_pair, 2, 1, 2))
        Z = rng.normal(size=(n_pair, 2))
        true_A = np.column_stack([0.4 + 0.2 * t, -0.3 + 0.1 * t])
        beta = np.asarray([0.5, -0.25])
        y = np.sum(X.reshape(n_pair, 2, 2) * true_A[:, None, :], axis=2)
        y += Z @ beta[:, None]
        y += rng.normal(scale=0.12, size=y.shape)
        dataset = PairedEyeDataset(
            subject_ids=np.asarray([f"pair_{idx}" for idx in range(n_pair)]),
            eye_ids=np.asarray(["OD", "OS"]),
            t=t,
            X=X,
            Z=Z,
            y=y,
        )
        config = {
            "z_mode": "full",
            "covariance_mode": "exchangeable_varying_sigma",
            "signal_h": 0.8,
            "variance_hbar": 0.8,
            "ridge": 1e-5,
        }
        t_grid = np.linspace(0.1, 0.9, num=9)
        A_hat, beta_hat, diagnostics = fit_coefficients_on_grid(dataset, config, t_grid)

        self.assertEqual(A_hat.shape, (9, 1, 2))
        self.assertEqual(beta_hat.shape, (2,))
        self.assertTrue(np.all(np.isfinite(A_hat)))
        self.assertTrue(np.all(np.isfinite(beta_hat)))
        self.assertGreater(diagnostics["sigma_min_eigenvalue"], 0.0)
        self.assertEqual(diagnostics["sigma2_hat_grid"].shape, (9,))
        self.assertEqual(diagnostics["local_support_pairs"].shape, (9,))
        self.assertEqual(diagnostics["variance_support_pairs"].shape, (9,))
        self.assertTrue(np.all(diagnostics["sigma2_hat_grid"] > 0.0))


class VarianceGridTest(unittest.TestCase):
    def test_variance_curve_accepts_arbitrary_grid(self) -> None:
        residual_pairs = np.asarray([[1.0, -1.0], [2.0, 0.0], [0.5, -0.5]])
        curve = smooth_variance_curve(
            residual_pairs=residual_pairs,
            t=np.asarray([0.0, 0.5, 1.0]),
            t_eval=np.asarray([0.25, 0.75]),
            bandwidth=0.8,
        )
        self.assertEqual(curve.shape, (2,))
        self.assertTrue(np.all(np.isfinite(curve)))
        self.assertTrue(np.all(curve > 0.0))


class PatientClusterResampleTest(unittest.TestCase):
    def test_resample_keeps_all_visits_and_assigns_unique_pair_ids(self) -> None:
        patient_ids = np.asarray([1, 1, 2, 3, 3, 3])
        pair_ids = np.asarray(["1_a", "1_b", "2_a", "3_a", "3_b", "3_c"])
        n_pair = patient_ids.size
        dataset = PairedEyeDataset(
            subject_ids=pair_ids.copy(),
            eye_ids=np.asarray(["OD", "OS"]),
            t=np.linspace(0.0, 1.0, num=n_pair),
            X=np.ones((n_pair, 2, 1, 1), dtype=float),
            Z=np.empty((n_pair, 0), dtype=float),
            y=np.ones((n_pair, 2), dtype=float),
        )
        manifest = pd.DataFrame({"subject_id": patient_ids, "pair_id": pair_ids})
        bootstrap, sampled, n_unique = patient_cluster_resample(
            dataset,
            manifest,
            np.random.default_rng(7),
            replicate=3,
        )

        expected_rows = sum(int(np.sum(patient_ids == patient_id)) for patient_id in sampled)
        self.assertEqual(bootstrap.n_subject, expected_rows)
        self.assertEqual(len(np.unique(bootstrap.subject_ids)), bootstrap.n_subject)
        self.assertEqual(n_unique, len(np.unique(sampled)))
        self.assertTrue(all(str(value).startswith("boot0003_draw") for value in bootstrap.subject_ids))

    def test_full_z_is_preserved_in_patient_resample(self) -> None:
        patient_ids = np.asarray([1, 1, 2, 3])
        pair_ids = np.asarray(["1_a", "1_b", "2_a", "3_a"])
        Z = np.arange(8, dtype=float).reshape(4, 2)
        dataset = PairedEyeDataset(
            subject_ids=pair_ids.copy(),
            eye_ids=np.asarray(["OD", "OS"]),
            t=np.linspace(0.0, 1.0, num=4),
            X=np.ones((4, 2, 1, 1), dtype=float),
            Z=Z,
            y=np.ones((4, 2), dtype=float),
        )
        manifest = pd.DataFrame({"subject_id": patient_ids, "pair_id": pair_ids})
        bootstrap, _, _ = patient_cluster_resample(
            dataset,
            manifest,
            np.random.default_rng(9),
            replicate=4,
            z_mode="full",
        )

        self.assertEqual(bootstrap.Z.shape[1], 2)
        self.assertTrue(all(any(np.array_equal(row, source) for source in Z) for row in bootstrap.Z))


class PairRowResampleTest(unittest.TestCase):
    def test_resample_keeps_od_os_pairs_and_fixed_row_count(self) -> None:
        pair_ids = np.asarray(["1_a", "1_b", "2_a", "3_a"])
        n_pair = pair_ids.size
        y = np.column_stack([np.arange(n_pair), np.arange(n_pair) + 100.0])
        dataset = PairedEyeDataset(
            subject_ids=pair_ids.copy(),
            eye_ids=np.asarray(["OD", "OS"]),
            t=np.linspace(0.0, 1.0, num=n_pair),
            X=np.ones((n_pair, 2, 1, 1), dtype=float),
            Z=np.arange(8, dtype=float).reshape(n_pair, 2),
            y=y,
        )
        manifest = pd.DataFrame({"subject_id": [1, 1, 2, 3], "pair_id": pair_ids})
        bootstrap, sampled, n_unique = pair_row_resample(
            dataset,
            manifest,
            np.random.default_rng(11),
            replicate=5,
            z_mode="selected",
        )

        self.assertEqual(bootstrap.n_subject, n_pair)
        self.assertEqual(bootstrap.Z.shape, (n_pair, 2))
        self.assertEqual(len(np.unique(bootstrap.subject_ids)), n_pair)
        self.assertEqual(n_unique, len(np.unique(sampled)))
        self.assertTrue(all(str(value).startswith("boot0005_draw") for value in bootstrap.subject_ids))
        np.testing.assert_allclose(bootstrap.y[:, 1] - bootstrap.y[:, 0], 100.0)


class SelectedZTest(unittest.TestCase):
    def test_selected_z_preserves_requested_order(self) -> None:
        names = ["is_female", "z_vf_01_mean", "z_vf_22_mean", "z_vf_31_mean"]
        Z = np.arange(20, dtype=float).reshape(5, 4)
        dataset = PairedEyeDataset(
            subject_ids=np.asarray([f"pair_{idx}" for idx in range(5)]),
            eye_ids=np.asarray(["OD", "OS"]),
            t=np.linspace(0.0, 1.0, num=5),
            X=np.ones((5, 2, 1, 1), dtype=float),
            Z=Z,
            y=np.ones((5, 2), dtype=float),
        )
        config = {
            "z_mode": "selected",
            "z_columns": ["is_female", "z_vf_31_mean", "z_vf_01_mean"],
        }
        selected, selected_names = select_z_columns(dataset, names, config)

        self.assertEqual(selected_names, config["z_columns"])
        np.testing.assert_array_equal(selected.Z, Z[:, [0, 3, 1]])


class JointBetaSummaryTest(unittest.TestCase):
    def test_significance_labels(self) -> None:
        self.assertEqual(significance_label(True, True), "both")
        self.assertEqual(significance_label(True, False), "CFP")
        self.assertEqual(significance_label(False, True), "ROI")
        self.assertEqual(significance_label(False, False), "none")


if __name__ == "__main__":
    unittest.main()

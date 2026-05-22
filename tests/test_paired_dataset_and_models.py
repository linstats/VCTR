from __future__ import annotations

import unittest

import numpy as np

from src.data import PairedEyeDataset
from src.dgps import PairedCase1DGP
from src.models import (
    InitialIidResult,
    PairedEyeVCTRModel,
    estimate_exchangeable_covariance,
    regroup_residuals_by_subject,
)


class PairedEyeDatasetTests(unittest.TestCase):
    def make_dataset(self) -> PairedEyeDataset:
        return PairedEyeDataset(
            subject_ids=np.array([10, 11, 12]),
            eye_ids=np.array([0, 1]),
            t=np.array([0.1, 0.4, 0.8]),
            Z=np.arange(6, dtype=float).reshape(3, 2),
            X=np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4),
            y=np.arange(6, dtype=float).reshape(3, 2),
            Sigma_true=np.array([[1.0, 0.25], [0.25, 1.0]]),
        )

    def test_valid_dataset_constructs(self) -> None:
        dataset = self.make_dataset()
        self.assertEqual(dataset.n_subject, 3)
        self.assertEqual(dataset.X.shape, (3, 2, 4))

    def test_invalid_eye_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            PairedEyeDataset(
                subject_ids=np.array([0, 1]),
                eye_ids=np.array([0, 1, 2]),
                t=np.array([0.1, 0.2]),
                Z=np.ones((2, 1)),
                X=np.ones((2, 2, 3)),
                y=np.ones((2, 2)),
            )

    def test_invalid_subject_eye_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            PairedEyeDataset(
                subject_ids=np.array([0, 1]),
                eye_ids=np.array([0, 1]),
                t=np.array([0.1]),
                Z=np.ones((2, 1)),
                X=np.ones((2, 2, 3)),
                y=np.ones((2, 2)),
            )

        with self.assertRaises(ValueError):
            PairedEyeDataset(
                subject_ids=np.array([0, 1]),
                eye_ids=np.array([0, 1]),
                t=np.array([0.1, 0.2]),
                Z=np.ones((2, 1)),
                X=np.ones((2, 3, 3)),
                y=np.ones((2, 2)),
            )

    def test_to_iid_observations_flattens_in_subject_major_eye_minor_order(self) -> None:
        dataset = self.make_dataset()
        flat = dataset.to_iid_observations()
        self.assertEqual(flat.X.shape, (6, 4))
        self.assertEqual(flat.y.shape, (6,))
        self.assertEqual(flat.t.shape, (6,))
        self.assertEqual(flat.Z.shape, (6, 2))
        np.testing.assert_array_equal(flat.subject_ids, np.array([10, 10, 11, 11, 12, 12]))
        np.testing.assert_array_equal(flat.eye_ids, np.array([0, 1, 0, 1, 0, 1]))
        np.testing.assert_array_equal(flat.y, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


class CovarianceHelperTests(unittest.TestCase):
    def test_regroup_residuals_by_subject(self) -> None:
        residual_pairs = regroup_residuals_by_subject(
            residuals=np.array([1.0, -1.0, 0.5, -0.5]),
            subject_ids=np.array([10, 10, 11, 11]),
            eye_ids=np.array([0, 1, 0, 1]),
        )
        np.testing.assert_allclose(residual_pairs, np.array([[1.0, -1.0], [0.5, -0.5]]))

    def test_regroup_residuals_duplicate_eye_raises(self) -> None:
        with self.assertRaises(ValueError):
            regroup_residuals_by_subject(
                residuals=np.array([1.0, 2.0]),
                subject_ids=np.array([10, 10]),
                eye_ids=np.array([0, 0]),
            )

    def test_estimate_exchangeable_covariance_returns_2x2_matrix(self) -> None:
        initial = InitialIidResult(
            residuals=np.array([1.0, 2.0, 0.0, 1.0], dtype=float),
            subject_ids=np.array([10, 10, 11, 11]),
            eye_ids=np.array([0, 1, 0, 1]),
        )
        estimate = estimate_exchangeable_covariance(initial)
        self.assertEqual(estimate.Sigma_hat.shape, (2, 2))
        self.assertEqual(estimate.residual_pairs.shape, (2, 2))

    def test_estimate_exchangeable_covariance_matches_equations_16_17(self) -> None:
        residuals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        initial = InitialIidResult(
            residuals=residuals,
            subject_ids=np.array([10, 10, 11, 11]),
            eye_ids=np.array([0, 1, 0, 1]),
        )
        estimate = estimate_exchangeable_covariance(initial)
        sigma2_expected = (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2) / 4.0
        rho_expected = ((1.0 * 2.0) + (3.0 * 4.0)) / 2.0 / sigma2_expected
        np.testing.assert_allclose(estimate.sigma2_hat, sigma2_expected)
        np.testing.assert_allclose(estimate.rho_hat, rho_expected)
        np.testing.assert_allclose(
            estimate.Sigma_hat,
            sigma2_expected * np.array([[1.0, rho_expected], [rho_expected, 1.0]]),
        )


class ModelWorkflowTests(unittest.TestCase):
    def make_dataset(self) -> PairedEyeDataset:
        return PairedEyeDataset(
            subject_ids=np.array([0, 1]),
            eye_ids=np.array([0, 1]),
            t=np.array([0.2, 0.7]),
            Z=np.ones((2, 1)),
            X=np.ones((2, 2, 3)),
            y=np.ones((2, 2)),
        )

    def test_fit_uses_three_stage_order(self) -> None:
        class DummyModel(PairedEyeVCTRModel):
            def __init__(self) -> None:
                super().__init__(bandwidth=0.1)
                self.calls: list[str] = []

            def initial_fit_iid(self, dataset: PairedEyeDataset) -> InitialIidResult:
                self.calls.append("initial")
                flat = dataset.to_iid_observations()
                return InitialIidResult(
                    fitted_values=np.zeros_like(flat.y),
                    residuals=np.array([1.0, 0.5, -1.0, -0.5]),
                    subject_ids=flat.subject_ids,
                    eye_ids=flat.eye_ids,
                )

            def estimate_covariance(self, dataset: PairedEyeDataset, initial_result: InitialIidResult):
                self.calls.append("covariance")
                return super().estimate_covariance(dataset, initial_result)

            def refit_with_covariance(self, dataset, covariance, initial_result=None):
                self.calls.append("refit")
                assert covariance.Sigma_hat.shape == (2, 2)
                from src.models import PairedVCTRResult

                return PairedVCTRResult(
                    initial=initial_result,
                    covariance=covariance,
                    fitted_values=np.zeros((dataset.n_subject, 2)),
                )

        model = DummyModel()
        result = model.fit(self.make_dataset())
        self.assertEqual(model.calls, ["initial", "covariance", "refit"])
        self.assertEqual(result.fitted_values.shape, (2, 2))

    def test_initial_and_weighted_fit_shapes(self) -> None:
        dataset = PairedEyeDataset(
            subject_ids=np.array([0, 1, 2]),
            eye_ids=np.array([0, 1]),
            t=np.array([0.2, 0.5, 0.8]),
            Z=np.ones((3, 1)),
            X=np.arange(3 * 2 * 2 * 2, dtype=float).reshape(3, 2, 2, 2) / 10.0,
            y=np.arange(6, dtype=float).reshape(3, 2) / 10.0,
        )
        model = PairedEyeVCTRModel(bandwidth=0.4, ridge=1e-4)
        initial = model.initial_fit_iid(dataset)
        self.assertEqual(initial.A_hat.shape, (3, 2, 2))
        self.assertEqual(initial.beta_hat.shape, (1,))
        self.assertEqual(initial.fitted_values.shape, (6,))
        self.assertEqual(initial.residuals.shape, (6,))

        covariance = model.estimate_covariance(dataset, initial)
        self.assertEqual(covariance.Sigma_hat.shape, (2, 2))

        final = model.refit_with_covariance(dataset, covariance, initial)
        self.assertEqual(final.A_hat.shape, (3, 2, 2))
        self.assertEqual(final.beta_hat.shape, (1,))
        self.assertEqual(final.fitted_values.shape, (3, 2))

    def test_fixed_bandwidth_does_not_trigger_auto_selection(self) -> None:
        class FixedBandwidthModel(PairedEyeVCTRModel):
            def _select_bandwidth_stage1_loo(self, dataset):
                raise AssertionError("auto-selection should not run in fixed mode")

        dataset = PairedCase1DGP(n_subject=8, R=2, S=2, p0=2).sample(seed=0)
        model = FixedBandwidthModel(bandwidth=0.13, ridge=1e-6)
        initial = model.initial_fit_iid(dataset)
        self.assertEqual(initial.meta["bandwidth_method"], "fixed")
        self.assertEqual(initial.meta["bandwidth_selected"], 0.13)
        self.assertEqual(initial.meta["bandwidth_grid"], [0.13])
        self.assertEqual(initial.meta["bandwidth_cv_scores"], [])

    def test_default_bandwidth_without_grid_falls_back_to_fixed_013(self) -> None:
        class DefaultBandwidthModel(PairedEyeVCTRModel):
            def _select_bandwidth_stage1_loo(self, dataset):
                raise AssertionError("auto-selection should not run without an explicit grid")

        dataset = PairedCase1DGP(n_subject=8, R=2, S=2, p0=2).sample(seed=3)
        model = DefaultBandwidthModel(bandwidth=None, bandwidth_grid=None, ridge=1e-6)
        initial = model.initial_fit_iid(dataset)
        self.assertEqual(initial.meta["bandwidth_method"], "default_fixed")
        self.assertEqual(initial.meta["bandwidth_selected"], 0.13)
        self.assertEqual(initial.meta["bandwidth_grid"], [0.13])
        self.assertEqual(initial.meta["bandwidth_cv_scores"], [])

    def test_auto_bandwidth_selects_from_grid_and_records_diagnostics(self) -> None:
        dataset = PairedCase1DGP(n_subject=8, R=2, S=2, p0=2).sample(seed=1)
        grid = (0.10, 0.13, 0.16)
        model = PairedEyeVCTRModel(
            bandwidth=None,
            bandwidth_method="stage1_kfold_cv",
            bandwidth_grid=grid,
            bandwidth_cv_folds=4,
            bandwidth_cv_seed=7,
            ridge=1e-6,
        )
        result = model.fit(dataset)
        selected = result.initial.meta["bandwidth_selected"]
        self.assertIn(selected, grid)
        self.assertEqual(result.initial.meta["bandwidth_grid"], list(grid))
        self.assertEqual(result.initial.meta["bandwidth_cv_metric"], "kfold_subject_mse")
        self.assertEqual(result.initial.meta["bandwidth_cv_folds"], 4)
        self.assertEqual(result.initial.meta["bandwidth_cv_seed"], 7)
        self.assertEqual(len(result.initial.meta["bandwidth_cv_scores"]), len(grid))
        self.assertEqual(result.meta["bandwidth_selected"], selected)

    def test_kfold_subject_cv_uses_requested_number_of_folds(self) -> None:
        class TrackingModel(PairedEyeVCTRModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fold_sizes: list[int] = []

            def _subject_fold_mse(self, dataset, bandwidth, holdout_indices):
                self.fold_sizes.append(len(holdout_indices))
                return float(len(holdout_indices))

        dataset = PairedCase1DGP(n_subject=10, R=2, S=2, p0=2).sample(seed=2)
        model = TrackingModel(
            bandwidth=None,
            bandwidth_method="stage1_kfold_cv",
            bandwidth_grid=(0.10, 0.13),
            bandwidth_cv_folds=5,
            ridge=1e-6,
        )
        _ = model._kfold_subject_cv_score(dataset, 0.10)
        self.assertEqual(len(model.fold_sizes), 5)
        self.assertEqual(sum(model.fold_sizes), 10)

    def test_loo_subject_cv_runs_one_fold_per_subject(self) -> None:
        class TrackingModel(PairedEyeVCTRModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.holdout_indices: list[int] = []

            def _loo_subject_fold_mse(self, dataset, bandwidth, holdout_index):
                self.holdout_indices.append(int(holdout_index))
                return float(holdout_index)

        dataset = PairedCase1DGP(n_subject=5, R=2, S=2, p0=2).sample(seed=2)
        model = TrackingModel(
            bandwidth=None,
            bandwidth_method="stage1_loo_cv",
            bandwidth_grid=(0.10, 0.13),
            ridge=1e-6,
        )
        _ = model._loo_subject_cv_score(dataset, 0.10)
        self.assertEqual(model.holdout_indices, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()

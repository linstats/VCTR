"""Tests for the fold-local GRAPE VF PCA transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.grape.evaluation.vf_pca_ablation import build_prediction_contrast_uncertainty
from src.experiments.grape.evaluation.vf_pca import (
    FoldVFPCATransformer,
    compose_pca_covariates,
    split_sex_vf,
    subject_equal_weights,
)


def test_subject_equal_weights_give_each_patient_equal_total_weight() -> None:
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    weights = subject_equal_weights(groups)
    totals = {group: float(weights[groups == group].sum()) for group in np.unique(groups)}
    assert totals == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_transformer_is_deterministic_and_training_centered() -> None:
    rng = np.random.default_rng(42)
    vf = rng.normal(size=(12, 5))
    groups = np.repeat(np.arange(6), 2)
    first = FoldVFPCATransformer.fit(vf, groups, n_components=3)
    second = FoldVFPCATransformer.fit(vf, groups, n_components=3)
    np.testing.assert_allclose(first.components_, second.components_)
    np.testing.assert_allclose(first.transform(vf), second.transform(vf))
    np.testing.assert_allclose(first.transform(vf), first.standardize(vf) @ first.components_.T)
    pivot_indices = np.argmax(np.abs(first.components_), axis=1)
    assert np.all(first.components_[np.arange(3), pivot_indices] >= 0)


def test_holdout_values_do_not_change_training_fit() -> None:
    rng = np.random.default_rng(9)
    train = rng.normal(size=(10, 4))
    groups = np.repeat(np.arange(5), 2)
    holdout_a = rng.normal(size=(3, 4))
    holdout_b = holdout_a + 1000.0
    transformer_a = FoldVFPCATransformer.fit(train, groups, n_components=2)
    transformer_b = FoldVFPCATransformer.fit(train, groups, n_components=2)
    np.testing.assert_allclose(transformer_a.mean_, transformer_b.mean_)
    np.testing.assert_allclose(transformer_a.scale_, transformer_b.scale_)
    np.testing.assert_allclose(transformer_a.components_, transformer_b.components_)
    assert not np.allclose(transformer_a.transform(holdout_a), transformer_b.transform(holdout_b))


def test_sex_is_kept_outside_pca_scores() -> None:
    Z = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]])
    sex, vf = split_sex_vf(Z)
    scores = np.array([[0.5], [-0.5]])
    np.testing.assert_array_equal(sex[:, 0], [0.0, 1.0])
    np.testing.assert_array_equal(vf, Z[:, 1:])
    np.testing.assert_array_equal(
        compose_pca_covariates(sex=sex, scores=scores, include_sex=True),
        np.array([[0.0, 0.5], [1.0, -0.5]]),
    )
    np.testing.assert_array_equal(
        compose_pca_covariates(sex=sex, scores=scores, include_sex=False),
        scores,
    )


def test_patient_cluster_prediction_contrast_uses_saved_oof_residuals() -> None:
    rows = []
    for subject_id in range(4):
        for eye in ("OD", "OS"):
            rows.extend(
                [
                    {
                        "image_type": "cfp",
                        "model": "x_only_paired_vctr",
                        "subject_id": subject_id,
                        "eye": eye,
                        "resid_iop": 2.0,
                    },
                    {
                        "image_type": "cfp",
                        "model": "x_vf_pca_paired_vctr",
                        "subject_id": subject_id,
                        "eye": eye,
                        "resid_iop": 1.0,
                    },
                ]
            )
    result = build_prediction_contrast_uncertainty(pd.DataFrame(rows), n_bootstrap=100, seed=7)
    row = result.loc[result["contrast"] == "adding_vf_pca_to_x"].iloc[0]
    assert row["delta_rmse_iop"] == -1.0
    assert row["ci_lower_delta_rmse_iop"] == -1.0
    assert row["ci_upper_delta_rmse_iop"] == -1.0
    assert row["bootstrap_probability_improvement"] == 1.0

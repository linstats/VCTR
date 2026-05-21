"""Reduced-scale MATLAB-style reproduction script for Case 3."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps import Case3BaselineDGP
from src.metrics import beta_mae, beta_rmse, npv, ppv, sensitivity, specificity, miae, rmise
from src.vctr.estimators import PenalizedSplineVCREstimator


@dataclass(slots=True)
class Case3MatlabRecord:
    rho: float
    rep: int
    seed: int
    miae: float
    rmise: float
    beta_mae: float
    beta_rmse: float
    se_beta_nonzero: float
    ppv_beta_nonzero: float
    spe_beta_nonzero: float
    npv_beta_nonzero: float
    se_const_zero: float
    ppv_const_zero: float
    spe_const_zero: float
    npv_const_zero: float
    se_const_nonzero: float
    ppv_const_nonzero: float
    spe_const_nonzero: float
    npv_const_nonzero: float
    se_vary: float
    ppv_vary: float
    spe_vary: float
    npv_vary: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output_dir = Path(__file__).resolve().parent / "results"
    parser.add_argument("--rho-values", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--n-rep", type=int, default=30)
    parser.add_argument("--penalty", type=str, default="SCAD", choices=["LASSO", "SCAD", "MCP"])
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--output-prefix", type=str, default="reproduce_case3_matlab")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))
    return parser.parse_args()


def case3_penalty_defaults(penalty: str, rho: float) -> tuple[float, float, float]:
    key = penalty.upper()
    if key == "LASSO":
        return 0.04, 0.02, 0.01
    if key == "SCAD":
        return 0.04, 0.02, 0.07
    if key == "MCP" and rho == 0.1:
        return 0.04, 0.02, 0.08
    if key == "MCP":
        return 0.04, 0.02, 0.09
    raise ValueError(f"Unsupported penalty: {penalty}")


def run(args: argparse.Namespace) -> tuple[list[Case3MatlabRecord], list[dict[str, float | int | str]]]:
    raw: list[Case3MatlabRecord] = []
    for rho in args.rho_values:
        lam_beta, lam_const, lam_vary = case3_penalty_defaults(args.penalty, rho)
        for rep in range(args.n_rep):
            seed = args.seed_base + rep
            dataset = Case3BaselineDGP(n=args.n, R=20, S=16, p0=5, sp=10, rho=rho, noise_scale=1.0).sample(seed=seed)
            result = PenalizedSplineVCREstimator(
                penalty=args.penalty,
                lambda_beta=lam_beta,
                lambda_const=lam_const,
                lambda_vary=lam_vary,
                threshold=args.threshold,
                ridge=args.ridge,
                max_iter=100,
                tol=1e-5,
                init_rank=2,
                init_max_iter=20,
                init_replicates=1,
                init_seed=seed,
            ).fit(dataset.X, dataset.Z, dataset.y, dataset.t)
            varying_true = dataset.meta["varying_mask_true"]
            const_zero_true = dataset.meta["const_zero_mask_true"]
            const_nonzero_true = dataset.meta["const_nonzero_mask_true"]
            beta_nonzero_true = dataset.meta["beta_nonzero_mask_true"]
            raw.append(
                Case3MatlabRecord(
                    rho=rho,
                    rep=rep,
                    seed=seed,
                    miae=miae(dataset.A_true, result.A_hat),
                    rmise=rmise(dataset.A_true, result.A_hat),
                    beta_mae=beta_mae(dataset.beta_true, result.beta_hat),
                    beta_rmse=beta_rmse(dataset.beta_true, result.beta_hat),
                    se_beta_nonzero=sensitivity(beta_nonzero_true, result.structure.beta_nonzero_mask),
                    ppv_beta_nonzero=ppv(beta_nonzero_true, result.structure.beta_nonzero_mask),
                    spe_beta_nonzero=specificity(beta_nonzero_true, result.structure.beta_nonzero_mask),
                    npv_beta_nonzero=npv(beta_nonzero_true, result.structure.beta_nonzero_mask),
                    se_const_zero=sensitivity(const_zero_true, result.structure.const_zero_mask),
                    ppv_const_zero=ppv(const_zero_true, result.structure.const_zero_mask),
                    spe_const_zero=specificity(const_zero_true, result.structure.const_zero_mask),
                    npv_const_zero=npv(const_zero_true, result.structure.const_zero_mask),
                    se_const_nonzero=sensitivity(const_nonzero_true, result.structure.const_nonzero_mask),
                    ppv_const_nonzero=ppv(const_nonzero_true, result.structure.const_nonzero_mask),
                    spe_const_nonzero=specificity(const_nonzero_true, result.structure.const_nonzero_mask),
                    npv_const_nonzero=npv(const_nonzero_true, result.structure.const_nonzero_mask),
                    se_vary=sensitivity(varying_true, result.structure.varying_mask),
                    ppv_vary=ppv(varying_true, result.structure.varying_mask),
                    spe_vary=specificity(varying_true, result.structure.varying_mask),
                    npv_vary=npv(varying_true, result.structure.varying_mask),
                )
            )
            print(f"[done] rho={rho:.1f} penalty={args.penalty:5s} rep={rep + 1}/{args.n_rep}")
    return raw, summarize_case3(raw)


def summarize_case3(records: Iterable[Case3MatlabRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[float, list[Case3MatlabRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.rho, []).append(rec)
    rows: list[dict[str, float | int | str]] = []
    fields = [f.name for f in Case3MatlabRecord.__dataclass_fields__.values() if f.name not in {"rho", "rep", "seed"}]
    for rho, vals in sorted(grouped.items()):
        row: dict[str, float | int | str] = {"rho": rho, "n_rep": len(vals)}
        for field in fields:
            arr = np.array([getattr(v, field) for v in vals], dtype=float)
            row[f"{field}_mean"] = float(np.mean(arr))
            row[f"{field}_std"] = float(np.std(arr, ddof=0))
        rows.append(row)
    return rows


def maybe_write_csv(raw: list[Case3MatlabRecord], summary: list[dict[str, float | int | str]], output_dir: str, output_prefix: str) -> None:
    if not output_prefix or not raw or not summary:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / f"{output_prefix}_raw.csv"
    summary_path = output_path / f"{output_prefix}_summary.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(raw[0]).keys()))
        writer.writeheader()
        for row in raw:
            writer.writerow(asdict(row))
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"Wrote raw results to {raw_path}")
    print(f"Wrote summary results to {summary_path}")


def print_summary(summary: list[dict[str, float | int | str]]) -> None:
    print("\nCase 3 MATLAB-style summary")
    for row in summary:
        print(
            f"rho={row['rho']:.1f}: "
            f"MIAE={row['miae_mean']:.4f} ({row['miae_std']:.4f}), "
            f"RMISE={row['rmise_mean']:.4f} ({row['rmise_std']:.4f}), "
            f"beta_MAE={row['beta_mae_mean']:.4f} ({row['beta_mae_std']:.4f}), "
            f"beta_RMSE={row['beta_rmse_mean']:.4f} ({row['beta_rmse_std']:.4f}), "
            f"vary_se={row['se_vary_mean']:.4f}, vary_ppv={row['ppv_vary_mean']:.4f}, "
            f"const1_se={row['se_const_nonzero_mean']:.4f}, beta_se={row['se_beta_nonzero_mean']:.4f}"
        )


def main() -> None:
    args = parse_args()
    raw, summary = run(args)
    print_summary(summary)
    maybe_write_csv(raw, summary, args.output_dir, args.output_prefix)


if __name__ == "__main__":
    main()

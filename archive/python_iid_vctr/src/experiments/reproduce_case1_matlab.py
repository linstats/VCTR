"""Reduced-scale MATLAB-style reproduction script for Case 1."""

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

from src.dgps import Case1BaselineDGP
from src.metrics import beta_mae, beta_rmse, mae, miae, rmise, rmse
from src.vctr.estimators import LocalLinearVCREstimator


DEFAULT_COEF_TYPES = ("sqrt", "quadratic", "bump", "sin")
DEFAULT_NOISE_TYPES = ("gaussian", "heavy_tailed")


@dataclass(slots=True)
class Case1MatlabRecord:
    n: int
    noise_type: str
    coef_type: str
    rep: int
    seed: int
    miae: float
    rmise: float
    beta_mae: float
    beta_rmse: float
    err_mae: float
    err_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output_dir = Path(__file__).resolve().parent / "results"
    parser.add_argument("--n-values", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--noise-types", type=str, nargs="+", default=list(DEFAULT_NOISE_TYPES))
    parser.add_argument("--coef-types", type=str, nargs="+", default=list(DEFAULT_COEF_TYPES))
    parser.add_argument("--n-rep", type=int, default=30)
    parser.add_argument("--bandwidth", type=float, default=0.13)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--output-prefix", type=str, default="reproduce_case1_matlab")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[list[Case1MatlabRecord], list[dict[str, float | int | str]]]:
    raw: list[Case1MatlabRecord] = []
    for n in args.n_values:
        for noise_type in args.noise_types:
            for coef_type in args.coef_types:
                for rep in range(args.n_rep):
                    seed = args.seed_base + rep
                    dataset = Case1BaselineDGP(
                        n=n,
                        R=10,
                        S=16,
                        p0=2,
                        coef_type=coef_type,
                        noise_type=noise_type,
                        beta_value=1.0,
                        noise_scale=1.0,
                    ).sample(seed=seed)
                    result = LocalLinearVCREstimator(
                        bandwidth=args.bandwidth,
                        ridge=args.ridge,
                        eval_mode="matlab_middle_random",
                        eval_fraction=0.2,
                        central_fraction=0.7,
                        eval_seed=seed,
                    ).fit(dataset.X, dataset.Z, dataset.y, dataset.t)
                    eval_indices = result.meta["eval_indices"]
                    A_true_eval = dataset.A_true[eval_indices]
                    y_true_eval = dataset.y[eval_indices]
                    raw.append(
                        Case1MatlabRecord(
                            n=n,
                            noise_type=noise_type,
                            coef_type=coef_type,
                            rep=rep,
                            seed=seed,
                            miae=miae(A_true_eval, result.A_hat),
                            rmise=rmise(A_true_eval, result.A_hat),
                            beta_mae=beta_mae(dataset.beta_true, result.beta_hat),
                            beta_rmse=beta_rmse(dataset.beta_true, result.beta_hat),
                            err_mae=mae(y_true_eval, result.fitted_values),
                            err_rmse=rmse(y_true_eval, result.fitted_values),
                        )
                    )
                    print(f"[done] n={n} noise={noise_type:12s} coef={coef_type:10s} rep={rep + 1}/{args.n_rep}")
    return raw, summarize(raw)


def summarize(records: Iterable[Case1MatlabRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str, str], list[Case1MatlabRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.n, rec.noise_type, rec.coef_type), []).append(rec)

    rows: list[dict[str, float | int | str]] = []
    for (n, noise_type, coef_type), vals in sorted(grouped.items()):
        row: dict[str, float | int | str] = {"n": n, "noise_type": noise_type, "coef_type": coef_type, "n_rep": len(vals)}
        for field in ("miae", "rmise", "beta_mae", "beta_rmse", "err_mae", "err_rmse"):
            arr = np.array([getattr(v, field) for v in vals], dtype=float)
            row[f"{field}_mean"] = float(np.mean(arr))
            row[f"{field}_std"] = float(np.std(arr, ddof=0))
        rows.append(row)
    return rows


def maybe_write_csv(raw: list[Case1MatlabRecord], summary: list[dict[str, float | int | str]], output_dir: str, output_prefix: str) -> None:
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
    print("\nCase 1 MATLAB-style summary")
    for row in summary:
        print(
            f"n={row['n']}, noise={row['noise_type']}, coef={row['coef_type']}: "
            f"MIAE={row['miae_mean']:.4f} ({row['miae_std']:.4f}), "
            f"RMISE={row['rmise_mean']:.4f} ({row['rmise_std']:.4f}), "
            f"beta_MAE={row['beta_mae_mean']:.4f} ({row['beta_mae_std']:.4f}), "
            f"beta_RMSE={row['beta_rmse_mean']:.4f} ({row['beta_rmse_std']:.4f}), "
            f"err_MAE={row['err_mae_mean']:.4f} ({row['err_mae_std']:.4f}), "
            f"err_RMSE={row['err_rmse_mean']:.4f} ({row['err_rmse_std']:.4f})"
        )


def main() -> None:
    args = parse_args()
    raw, summary = run(args)
    print_summary(summary)
    maybe_write_csv(raw, summary, args.output_dir, args.output_prefix)


if __name__ == "__main__":
    main()

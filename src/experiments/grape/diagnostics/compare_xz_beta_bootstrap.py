"""Compare CFP/ROI X+Z beta bootstrap summaries and retain variables significant in either model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.grape.diagnostics.bootstrap_coefficients import (  # noqa: E402
    GRAPE_ROOT,
    RUN_ROOT,
    rel_to_repo,
    resolve_path,
)


DEFAULT_CFP_RUN = RUN_ROOT / "cfp_xz_inherit_xonly_tuning_b2000"
DEFAULT_ROI_RUN = RUN_ROOT / "roi_xz_inherit_xonly_tuning_b2000"
DEFAULT_RUN_NAME = "xz_inherit_xonly_tuning_b2000_comparison"
DEFAULT_OUTPUT_ROOT = GRAPE_ROOT / "outputs" / "coefficient_bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfp-run", type=Path, default=DEFAULT_CFP_RUN)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def load_beta_summary(run_dir: Path, expected_image: str) -> pd.DataFrame:
    path = run_dir / "beta_summary_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing beta summary: {path}")
    frame = pd.read_csv(path)
    required = {
        "image_type",
        "variable",
        "variable_type",
        "beta_hat_iop",
        "bootstrap_se_iop",
        "ci_lower_iop",
        "ci_upper_iop",
        "ci_excludes_zero",
        "n_success",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    images = set(frame["image_type"].astype(str).str.lower())
    if images != {expected_image}:
        raise ValueError(f"Expected image_type={expected_image!r} in {path}; got {sorted(images)}")
    if frame["variable"].duplicated().any():
        raise ValueError(f"Duplicate variables in {path}")
    return frame


def prefixed_beta_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    shared = frame[["variable", "variable_type"]].copy()
    fields = (
        "beta_hat_std",
        "bootstrap_se_std",
        "ci_lower_std",
        "ci_upper_std",
        "beta_hat_iop",
        "bootstrap_se_iop",
        "ci_lower_iop",
        "ci_upper_iop",
        "ci_excludes_zero",
        "n_success",
    )
    for field in fields:
        shared[f"{prefix}_{field}"] = frame[field].to_numpy()
    return shared


def significance_label(cfp: bool, roi: bool) -> str:
    if cfp and roi:
        return "both"
    if cfp:
        return "CFP"
    if roi:
        return "ROI"
    return "none"


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    cfp_run = resolve_path(args.cfp_run)
    roi_run = resolve_path(args.roi_run)
    run_dir = resolve_path(args.run_root) / str(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfp = prefixed_beta_columns(load_beta_summary(cfp_run, "cfp"), "cfp")
    roi = prefixed_beta_columns(load_beta_summary(roi_run, "roi"), "roi")
    combined = cfp.merge(roi, on=["variable", "variable_type"], how="outer", validate="one_to_one")
    if combined.isna().any().any():
        raise ValueError("CFP and ROI beta summaries must contain the same complete variable set.")
    combined["cfp_significant"] = combined["cfp_ci_excludes_zero"].astype(bool)
    combined["roi_significant"] = combined["roi_ci_excludes_zero"].astype(bool)
    combined["significant_either_image"] = combined["cfp_significant"] | combined["roi_significant"]
    combined["significant_in"] = [
        significance_label(cfp_sig, roi_sig)
        for cfp_sig, roi_sig in zip(
            combined["cfp_significant"],
            combined["roi_significant"],
            strict=True,
        )
    ]
    combined = combined.sort_values(["variable_type", "variable"], kind="mergesort").reset_index(drop=True)
    significant = combined[combined["significant_either_image"]].copy()

    all_path = run_dir / "beta_summary_all_images.csv"
    significant_path = run_dir / "beta_summary_significant_either_image.csv"
    combined.to_csv(all_path, index=False)
    significant.to_csv(significant_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cfp_run": rel_to_repo(cfp_run),
        "roi_run": rel_to_repo(roi_run),
        "run_dir": rel_to_repo(run_dir),
        "n_variables_total": int(len(combined)),
        "n_variables_significant_either_image": int(len(significant)),
        "selection_rule": "CFP or ROI nominal 95% percentile bootstrap CI excludes zero",
        "multiple_testing_adjustment": "none",
        "interpretation_warning": "Filtered variables are nominally significant; 120 image-by-variable comparisons are not multiplicity-adjusted.",
    }
    (run_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        f"# {args.run_name}",
        "",
        "Joint CFP/ROI regression-coefficient summary for X+Z paired VCTR models using inherited X-only tuning.",
        "",
        "The manuscript-facing table retains a variable when its nominal 95% percentile bootstrap CI excludes zero in CFP or ROI.",
        "The complete 60-variable table remains available for audit. No multiple-testing adjustment is applied.",
        "",
        "## Files",
        "",
        "- `beta_summary_all_images.csv`: complete CFP/ROI audit table",
        "- `beta_summary_significant_either_image.csv`: filtered presentation table",
        "- `comparison_metadata.json`: provenance and filtering rule",
    ]
    (run_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    if not args.no_export:
        output_dir = resolve_path(args.output_root) / "xz_inherit_xonly_tuning_b2000"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(significant_path, output_dir / significant_path.name)
        shutil.copy2(run_dir / "comparison_metadata.json", output_dir / "comparison_metadata.json")
        shutil.copy2(run_dir / "README.md", output_dir / "README.md")
        for image, source_run in (("cfp", cfp_run), ("roi", roi_run)):
            figure_dir = source_run / "figures"
            copy_if_exists(figure_dir / f"{image}_at_pointwise_ci.png", output_dir / f"{image}_at_pointwise_ci.png")
            copy_if_exists(figure_dir / f"{image}_at_pointwise_ci.pdf", output_dir / f"{image}_at_pointwise_ci.pdf")
        metadata["output_dir"] = rel_to_repo(output_dir)

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

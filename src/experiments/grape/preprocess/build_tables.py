"""Build GRAPE interim and paired processed tables from the raw workbook.

This script keeps the raw Excel and images unchanged. It creates:

- src/experiments/grape/data/audit/interim_visits.csv: one row per eye visit.
- src/experiments/grape/data/audit/processed_paired.csv: one row per paired OD/OS visit.
- src/experiments/grape/data/audit/build_summary.json: compact audit counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


GRAPE_ROOT = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = GRAPE_ROOT / "raw"
AUDIT_DIR = GRAPE_ROOT / "audit"
WORKBOOK = RAW_DIR / "VF_and_clinical_information.xlsx"
CFP_DIR = RAW_DIR / "CFPs"
ROI_DIR = RAW_DIR / "ROIs"

INTERIM_CSV = AUDIT_DIR / "interim_visits.csv"
PROCESSED_CSV = AUDIT_DIR / "processed_paired.csv"
SUMMARY_JSON = AUDIT_DIR / "build_summary.json"

BLIND_SPOT_VF = {21, 32}


def _rel_to_grape_data(path: Path) -> str:
    """Return a portable path relative to the GRAPE data directory."""

    return path.relative_to(GRAPE_ROOT).as_posix()


def _clean_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _read_grape_sheet(sheet_name: str) -> pd.DataFrame:
    """Read a GRAPE sheet with the original two-row merged header."""

    raw = pd.read_excel(WORKBOOK, sheet_name=sheet_name, header=None)
    header_top = raw.iloc[0].ffill()
    header_sub = raw.iloc[1]
    columns: list[str] = []
    for top, sub in zip(header_top, header_sub, strict=True):
        top_name = str(top).strip()
        if top_name == "VF":
            columns.append(f"vf_{int(sub):02d}")
        elif top_name == "Progression Status":
            columns.append(f"progression_{_clean_name(sub)}")
        elif top_name == "OCT RNFL thickness":
            columns.append(f"rnfl_{_clean_name(sub)}")
        else:
            columns.append(_clean_name(top_name))

    data = raw.iloc[2:].copy()
    data.columns = columns
    return data.reset_index(drop=True)


def _normalize_visit_table(followup: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Create the visit-level fact table."""

    baseline_static_cols = [
        "subject_number",
        "laterality",
        "age",
        "gender",
        "cct",
        "total_visits",
        "progression_plr2",
        "progression_plr3",
        "progression_md",
        "category_of_glaucoma",
        "rnfl_mean",
        "rnfl_s",
        "rnfl_n",
        "rnfl_i",
        "rnfl_t",
    ]
    baseline_static = baseline[baseline_static_cols].copy()
    baseline_static = baseline_static.rename(
        columns={
            "subject_number": "subject_id",
            "age": "baseline_age",
            "cct": "baseline_cct",
            "total_visits": "baseline_total_visits",
            "category_of_glaucoma": "baseline_glaucoma_category",
        }
    )
    baseline_static["subject_id"] = baseline_static["subject_id"].astype(int)

    visits = followup.copy()
    visits = visits.rename(
        columns={
            "subject_number": "subject_id",
            "resolusion": "resolution",
        }
    )
    visits["subject_id"] = visits["subject_id"].astype(int)
    visits["visit_number"] = visits["visit_number"].astype(int)
    visits["interval_years"] = visits["interval_years"].astype(float)
    visits["iop"] = visits["iop"].astype(float)

    visits = visits.merge(
        baseline_static,
        on=["subject_id", "laterality"],
        how="left",
        validate="many_to_one",
    )

    visits["eye_id"] = visits["subject_id"].astype(str) + "_" + visits["laterality"].astype(str)
    visits["observation_id"] = (
        visits["subject_id"].astype(str)
        + "_"
        + visits["laterality"].astype(str)
        + "_"
        + visits["visit_number"].astype(str)
    )
    visits["age_at_visit"] = visits["baseline_age"].astype(float) + visits["interval_years"]
    visits["is_female"] = (visits["gender"] == "F").astype(int)

    cfp_name = visits["corresponding_cfp"].astype(str)
    visits["has_cfp"] = visits["corresponding_cfp"].notna() & (cfp_name != "/")
    visits["cfp_path"] = ""
    visits["roi_path"] = ""
    visits.loc[visits["has_cfp"], "cfp_path"] = cfp_name[visits["has_cfp"]].map(
        lambda x: _rel_to_grape_data(CFP_DIR / x)
    )
    visits.loc[visits["has_cfp"], "roi_path"] = cfp_name[visits["has_cfp"]].map(
        lambda x: _rel_to_grape_data(ROI_DIR / x)
    )
    visits["cfp_exists"] = visits["cfp_path"].map(lambda p: (GRAPE_ROOT / p).exists() if p else False)
    visits["roi_exists"] = visits["roi_path"].map(lambda p: (GRAPE_ROOT / p).exists() if p else False)

    image_visits = visits[visits["has_cfp"]].copy()
    iop_mean = float(image_visits["iop"].mean())
    iop_sd = float(image_visits["iop"].std(ddof=1))
    visits["iop_outlier_row"] = visits["has_cfp"] & (
        (visits["iop"] > iop_mean + 2 * iop_sd) | (visits["iop"] < iop_mean - 2 * iop_sd)
    )
    visits["iop_gt35_visit"] = visits["has_cfp"] & (visits["iop"] > 35)
    visits["iop_eq7_or_gt30_visit"] = visits["has_cfp"] & ((visits["iop"] == 7) | (visits["iop"] > 30))
    visits["include_primary_iop35"] = ~visits["iop_gt35_visit"]
    visits["include_sensitivity_iop30_low7"] = ~visits["iop_eq7_or_gt30_visit"]
    outlier_eyes = visits.loc[visits["iop_outlier_row"], ["subject_id", "laterality"]].drop_duplicates()
    outlier_eyes["eye_has_iop_outlier"] = True
    visits = visits.merge(outlier_eyes, on=["subject_id", "laterality"], how="left")
    visits["eye_has_iop_outlier"] = visits["eye_has_iop_outlier"].eq(True)

    ordered_cols = [
        "observation_id",
        "subject_id",
        "laterality",
        "eye_id",
        "visit_number",
        "interval_years",
        "baseline_age",
        "age_at_visit",
        "gender",
        "is_female",
        "iop",
        "corresponding_cfp",
        "cfp_path",
        "roi_path",
        "has_cfp",
        "cfp_exists",
        "roi_exists",
        "iop_outlier_row",
        "iop_gt35_visit",
        "iop_eq7_or_gt30_visit",
        "include_primary_iop35",
        "include_sensitivity_iop30_low7",
        "eye_has_iop_outlier",
        "acquisition_device",
        "resolution",
        "baseline_cct",
        "baseline_total_visits",
        "progression_plr2",
        "progression_plr3",
        "progression_md",
        "baseline_glaucoma_category",
        "rnfl_mean",
        "rnfl_s",
        "rnfl_n",
        "rnfl_i",
        "rnfl_t",
    ]
    vf_cols = [f"vf_{i:02d}" for i in range(61)]
    return visits[ordered_cols + vf_cols].sort_values(["subject_id", "laterality", "visit_number"])


def _build_paired_table(visits: pd.DataFrame) -> pd.DataFrame:
    """Build one row per OD/OS pair at the same subject and interval time."""

    image_visits = visits[visits["has_cfp"]].copy()
    od = image_visits[image_visits["laterality"] == "OD"].copy()
    os = image_visits[image_visits["laterality"] == "OS"].copy()

    paired = od.merge(
        os,
        on=["subject_id", "interval_years"],
        suffixes=("_od", "_os"),
        how="inner",
        validate="one_to_one",
    )
    paired["pair_id"] = paired["subject_id"].astype(str) + "_" + paired["interval_years"].map(
        lambda x: f"{float(x):.12g}"
    )
    paired["age_at_visit"] = paired["age_at_visit_od"]
    paired["gender"] = paired["gender_od"]
    paired["is_female"] = paired["is_female_od"]
    paired["pair_has_iop_outlier"] = paired["eye_has_iop_outlier_od"] | paired["eye_has_iop_outlier_os"]
    paired["include_old_iop_rule"] = ~paired["pair_has_iop_outlier"]
    paired["pair_has_iop_gt35_visit"] = paired["iop_gt35_visit_od"] | paired["iop_gt35_visit_os"]
    paired["pair_has_iop_eq7_or_gt30_visit"] = (
        paired["iop_eq7_or_gt30_visit_od"] | paired["iop_eq7_or_gt30_visit_os"]
    )
    paired["include_primary_iop35"] = (
        paired["include_primary_iop35_od"] & paired["include_primary_iop35_os"]
    )
    paired["include_sensitivity_iop30_low7"] = (
        paired["include_sensitivity_iop30_low7_od"] & paired["include_sensitivity_iop30_low7_os"]
    )

    base_cols = [
        "pair_id",
        "subject_id",
        "interval_years",
        "age_at_visit",
        "gender",
        "is_female",
        "visit_number_od",
        "visit_number_os",
        "observation_id_od",
        "observation_id_os",
        "iop_od",
        "iop_os",
        "corresponding_cfp_od",
        "corresponding_cfp_os",
        "cfp_path_od",
        "cfp_path_os",
        "roi_path_od",
        "roi_path_os",
        "cfp_exists_od",
        "cfp_exists_os",
        "roi_exists_od",
        "roi_exists_os",
        "iop_outlier_row_od",
        "iop_outlier_row_os",
        "iop_gt35_visit_od",
        "iop_gt35_visit_os",
        "iop_eq7_or_gt30_visit_od",
        "iop_eq7_or_gt30_visit_os",
        "eye_has_iop_outlier_od",
        "eye_has_iop_outlier_os",
        "pair_has_iop_outlier",
        "include_old_iop_rule",
        "pair_has_iop_gt35_visit",
        "include_primary_iop35",
        "pair_has_iop_eq7_or_gt30_visit",
        "include_sensitivity_iop30_low7",
    ]

    derived_cols: dict[str, pd.Series] = {}
    for idx in range(61):
        col = f"vf_{idx:02d}"
        if idx not in BLIND_SPOT_VF:
            derived_cols[f"z_vf_{idx:02d}_mean"] = paired[[f"{col}_od", f"{col}_os"]].astype(float).mean(axis=1)

    paired = pd.concat([paired, pd.DataFrame(derived_cols, index=paired.index)], axis=1)

    vf_pair_cols = [f"vf_{i:02d}_{eye}" for i in range(61) for eye in ("od", "os")]
    z_cols = [f"z_vf_{i:02d}_mean" for i in range(61) if i not in BLIND_SPOT_VF]
    return paired[base_cols + vf_pair_cols + z_cols].sort_values(["subject_id", "interval_years"])


def _build_summary(visits: pd.DataFrame, paired: pd.DataFrame) -> dict[str, object]:
    image_visits = visits[visits["has_cfp"]]
    paired_observation_ids = set(paired["observation_id_od"]).union(set(paired["observation_id_os"]))
    unpaired = image_visits[~image_visits["observation_id"].isin(paired_observation_ids)]
    never_paired_physical_eyes = (
        image_visits[["subject_id", "laterality"]]
        .drop_duplicates()
        .merge(
            image_visits[image_visits["observation_id"].isin(paired_observation_ids)][
                ["subject_id", "laterality"]
            ].drop_duplicates(),
            on=["subject_id", "laterality"],
            how="left",
            indicator=True,
        )
    )
    never_paired_physical_eyes = never_paired_physical_eyes[
        never_paired_physical_eyes["_merge"] == "left_only"
    ]

    return {
        "raw_workbook": _rel_to_grape_data(WORKBOOK),
        "n_visits": int(len(visits)),
        "n_subjects": int(visits["subject_id"].nunique()),
        "n_physical_eyes": int(visits[["subject_id", "laterality"]].drop_duplicates().shape[0]),
        "n_visits_with_cfp": int(len(image_visits)),
        "n_cfp_files": int(len(list(CFP_DIR.glob("*.jpg")))),
        "n_roi_files": int(len(list(ROI_DIR.glob("*.jpg")))),
        "n_image_visits_missing_cfp_file": int((~image_visits["cfp_exists"]).sum()),
        "n_image_visits_missing_roi_file": int((~image_visits["roi_exists"]).sum()),
        "n_paired_rows_same_interval": int(len(paired)),
        "n_paired_eye_visits": int(len(paired) * 2),
        "n_unpaired_eye_visits": int(len(unpaired)),
        "n_unpaired_eye_visits_by_laterality": {
            str(k): int(v) for k, v in unpaired["laterality"].value_counts().sort_index().items()
        },
        "n_physical_eyes_never_paired": int(len(never_paired_physical_eyes)),
        "n_pairs_include_old_iop_rule": int(paired["include_old_iop_rule"].sum()),
        "n_pairs_excluded_by_old_iop_rule": int((~paired["include_old_iop_rule"]).sum()),
        "n_image_visits_include_primary_iop35": int(image_visits["include_primary_iop35"].sum()),
        "n_image_visits_excluded_by_primary_iop35": int((~image_visits["include_primary_iop35"]).sum()),
        "n_pairs_include_primary_iop35": int(paired["include_primary_iop35"].sum()),
        "n_pairs_excluded_by_primary_iop35": int((~paired["include_primary_iop35"]).sum()),
        "n_image_visits_include_sensitivity_iop30_low7": int(
            image_visits["include_sensitivity_iop30_low7"].sum()
        ),
        "n_image_visits_excluded_by_sensitivity_iop30_low7": int(
            (~image_visits["include_sensitivity_iop30_low7"]).sum()
        ),
        "n_pairs_include_sensitivity_iop30_low7": int(paired["include_sensitivity_iop30_low7"].sum()),
        "n_pairs_excluded_by_sensitivity_iop30_low7": int(
            (~paired["include_sensitivity_iop30_low7"]).sum()
        ),
        "blind_spot_vf_columns": sorted(BLIND_SPOT_VF),
        "z_vf_mean_columns": int(61 - len(BLIND_SPOT_VF)),
    }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _read_grape_sheet("Baseline")
    followup = _read_grape_sheet("Follow-up")
    visits = _normalize_visit_table(followup=followup, baseline=baseline)
    paired = _build_paired_table(visits)
    summary = _build_summary(visits, paired)

    visits.to_csv(INTERIM_CSV, index=False)
    paired.to_csv(PROCESSED_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

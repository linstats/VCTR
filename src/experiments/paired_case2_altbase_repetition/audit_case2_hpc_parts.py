"""Audit incomplete HPC Case 2 part runs and generate exact backfill manifests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


EXPECTED_MISSING_TOTAL = 213
MANIFEST_FIELDS = ("part", "n_subject", "coef_type", "rho_true", "rep", "seed")


@dataclass(frozen=True, slots=True)
class TaskKey:
    n_subject: int
    coef_type: str
    rho_true: float
    rep: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        required=True,
        help="Snapshot root containing part1/ ... part8/ copied from HPC.",
    )
    parser.add_argument(
        "--expected-parts",
        type=int,
        default=8,
        help="Number of part directories expected under the snapshot root.",
    )
    parser.add_argument(
        "--expected-missing-total",
        type=int,
        default=EXPECTED_MISSING_TOTAL,
        help="Expected total missing-task count for validation.",
    )
    parser.add_argument(
        "--remote-project-root",
        type=str,
        default="$HOME/2026-tensor",
        help="Remote project root used when generating sync/qsub helper scripts.",
    )
    parser.add_argument(
        "--remote-manifest-dir",
        type=str,
        default="$HOME/2026-tensor/hpc/manifests/case2_backfill_20260529",
        help="Remote directory where manifest CSVs will be uploaded.",
    )
    parser.add_argument(
        "--remote-user-host",
        type=str,
        default="e0829076@atlas9.nus.edu.sg",
        help="Remote user@host used in generated scp helper commands.",
    )
    parser.add_argument(
        "--conda-module",
        type=str,
        default="miniconda/4.12",
        help="Module name written into generated qsub commands.",
    )
    parser.add_argument(
        "--conda-env-path",
        type=str,
        default="$HOME/conda-envs/vctr-py310",
        help="Conda env path written into generated qsub commands.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=12,
        help="Default worker count written into generated qsub commands.",
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="run_case2_altbase_backfill",
        help="Prefix for generated backfill run names.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def float_label(value: float) -> str:
    return format(float(value), ".12g")


def canonical_key(row: dict[str, str]) -> TaskKey:
    return TaskKey(
        n_subject=int(row["n_subject"]),
        coef_type=str(row["coef_type"]),
        rho_true=float(row["rho_true"]),
        rep=int(row["rep"]),
        seed=int(row["seed"]),
    )


def expected_tasks(config: dict) -> list[TaskKey]:
    rho_values = config.get("rho_values")
    if not rho_values:
        rho_values = [config["rho"]]
    seed_base = int(config["seed_base"])
    tasks: list[TaskKey] = []
    for n_subject in config["n_subject_values"]:
        for coef_type in config["coef_types"]:
            for rho_true in rho_values:
                for rep in range(int(config["n_rep"])):
                    tasks.append(
                        TaskKey(
                            n_subject=int(n_subject),
                            coef_type=str(coef_type),
                            rho_true=float(rho_true),
                            rep=rep,
                            seed=seed_base + rep,
                        )
                    )
    return tasks


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_by_n_subject(keys: Iterable[TaskKey]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        label = str(key.n_subject)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def count_by_group(keys: Iterable[TaskKey]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        label = f"{key.coef_type}|rho={float_label(key.rho_true)}"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def shell_quote(path: Path | str) -> str:
    text = str(path)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def build_sync_script(
    *,
    audit_dir: Path,
    snapshot_root: Path,
    remote_user_host: str,
    remote_project_root: str,
    remote_manifest_dir: str,
) -> str:
    local_root = Path(__file__).resolve().parents[3]
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"REMOTE={shell_quote(remote_user_host)}",
        f"REMOTE_PROJECT_ROOT={remote_project_root}",
        f"REMOTE_MANIFEST_DIR={remote_manifest_dir}",
        "",
        'scp '
        + shell_quote(local_root / "src/experiments/paired_case2_altbase_repetition/audit_case2_hpc_parts.py")
        + ' "$REMOTE:$REMOTE_PROJECT_ROOT/src/experiments/paired_case2_altbase_repetition/"',
        'scp '
        + shell_quote(local_root / "src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_backfill.py")
        + ' "$REMOTE:$REMOTE_PROJECT_ROOT/src/experiments/paired_case2_altbase_repetition/"',
        'scp '
        + shell_quote(local_root / "hpc/paired_case2_altbase_backfill_parallel.pbs")
        + ' "$REMOTE:$REMOTE_PROJECT_ROOT/hpc/"',
        'ssh "$REMOTE" "mkdir -p $REMOTE_MANIFEST_DIR"',
    ]
    for part_path in sorted(audit_dir.glob("part*_missing.csv")):
        lines.append(
            "scp "
            + shell_quote(part_path)
            + ' "$REMOTE:$REMOTE_MANIFEST_DIR/"'
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_submit_script(
    *,
    audit_dir: Path,
    remote_project_root: str,
    remote_manifest_dir: str,
    conda_module: str,
    conda_env_path: str,
    n_jobs: int,
    run_name_prefix: str,
) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"PROJECT_ROOT={remote_project_root}",
        f"MANIFEST_DIR={remote_manifest_dir}",
        "",
    ]
    for manifest_path in sorted(audit_dir.glob("part*_missing.csv")):
        part_name = manifest_path.stem.replace("_missing", "")
        run_name = f"{run_name_prefix}_{part_name}_20260529"
        cmd = (
            "qsub -v "
            f"PROJECT_ROOT=$PROJECT_ROOT,"
            f"CONDA_MODULE={conda_module},"
            f"CONDA_ENV_PATH={conda_env_path},"
            f"N_JOBS={n_jobs},"
            f"MANIFEST_PATH=$MANIFEST_DIR/{manifest_path.name},"
            f"RUN_NAME={run_name} "
            "hpc/paired_case2_altbase_backfill_parallel.pbs"
        )
        lines.append(cmd)
    lines.append("")
    return "\n".join(lines)


def build_readme(
    *,
    snapshot_root: Path,
    expected_parts: int,
    total_missing: int,
    expected_missing_total: int,
    part_summaries: list[dict],
) -> str:
    lines = [
        f"Snapshot root: {snapshot_root}",
        f"Expected parts: {expected_parts}",
        f"Audited parts: {len(part_summaries)}",
        f"Total missing tasks: {total_missing}",
        f"Expected missing total: {expected_missing_total}",
        "",
    ]
    for summary in part_summaries:
        lines.append(
            "part{part}: expected={expected_total} success={success_count} fail={failure_count} "
            "duplicate_success={duplicate_success_count} missing={missing_count}".format(**summary)
        )
        lines.append(f"  missing_by_n_subject={summary['missing_by_n_subject']}")
        lines.append(f"  missing_by_group={summary['missing_by_group']}")
    lines.append("")
    if total_missing != expected_missing_total:
        lines.append("WARNING: total missing count does not match the expected benchmark.")
    all_missing_n = {
        n_subject
        for summary in part_summaries
        for n_subject, count in summary["missing_by_n_subject"].items()
        if count > 0
    }
    lines.append(f"Missing n_subject values observed: {sorted(all_missing_n, key=int)}")
    return "\n".join(lines) + "\n"


def audit_part(part: int, part_dir: Path) -> tuple[list[dict[str, object]], dict]:
    config_path = part_dir / "run_config.json"
    raw_path = part_dir / "results" / "raw_results.csv"
    config = load_json(config_path)
    rows = read_rows(raw_path)

    expected = expected_tasks(config)
    expected_set = set(expected)

    success_keys: list[TaskKey] = []
    failure_rows: list[dict[str, str]] = []
    duplicate_success: list[TaskKey] = []
    seen_success: set[TaskKey] = set()
    invalid_success: list[TaskKey] = []

    for row in rows:
        key = canonical_key(row)
        if int(row["success"]) == 1:
            if key in seen_success:
                duplicate_success.append(key)
            else:
                seen_success.add(key)
                success_keys.append(key)
            if key not in expected_set:
                invalid_success.append(key)
        else:
            failure_rows.append(row)

    success_set = set(success_keys)
    missing = sorted(expected_set - success_set, key=lambda item: (item.n_subject, item.coef_type, item.rho_true, item.rep, item.seed))
    missing_rows = [
        {
            "part": part,
            "n_subject": key.n_subject,
            "coef_type": key.coef_type,
            "rho_true": float_label(key.rho_true),
            "rep": key.rep,
            "seed": key.seed,
        }
        for key in missing
    ]

    summary = {
        "part": part,
        "config_path": str(config_path),
        "raw_results_path": str(raw_path),
        "expected_total": len(expected),
        "rows_total": len(rows),
        "success_count": len(success_keys),
        "failure_count": len(failure_rows),
        "duplicate_success_count": len(duplicate_success),
        "invalid_success_count": len(invalid_success),
        "missing_count": len(missing_rows),
        "missing_by_n_subject": count_by_n_subject(missing),
        "missing_by_group": count_by_group(missing),
        "failure_examples": failure_rows[:5],
        "duplicate_success_examples": [
            {
                "n_subject": item.n_subject,
                "coef_type": item.coef_type,
                "rho_true": float_label(item.rho_true),
                "rep": item.rep,
                "seed": item.seed,
            }
            for item in duplicate_success[:5]
        ],
        "invalid_success_examples": [
            {
                "n_subject": item.n_subject,
                "coef_type": item.coef_type,
                "rho_true": float_label(item.rho_true),
                "rep": item.rep,
                "seed": item.seed,
            }
            for item in invalid_success[:5]
        ],
    }
    return missing_rows, summary


def main() -> None:
    args = parse_args()
    snapshot_root = args.snapshot_root.resolve()
    audit_dir = snapshot_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    all_missing_rows: list[dict[str, object]] = []
    part_summaries: list[dict] = []

    for part in range(1, args.expected_parts + 1):
        part_dir = snapshot_root / f"part{part}"
        if not part_dir.exists():
            raise FileNotFoundError(f"missing snapshot directory: {part_dir}")
        missing_rows, summary = audit_part(part, part_dir)
        write_csv(audit_dir / f"part{part}_missing.csv", missing_rows, MANIFEST_FIELDS)
        with (audit_dir / f"part{part}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        all_missing_rows.extend(missing_rows)
        part_summaries.append(summary)

    write_csv(audit_dir / "all_missing.csv", all_missing_rows, MANIFEST_FIELDS)

    aggregate = {
        "snapshot_root": str(snapshot_root),
        "audit_dir": str(audit_dir),
        "expected_parts": args.expected_parts,
        "audited_parts": len(part_summaries),
        "expected_missing_total": args.expected_missing_total,
        "total_missing": len(all_missing_rows),
        "all_missing_by_n_subject": {},
        "part_summaries": part_summaries,
    }
    all_missing_keys = [
        TaskKey(
            n_subject=int(row["n_subject"]),
            coef_type=str(row["coef_type"]),
            rho_true=float(row["rho_true"]),
            rep=int(row["rep"]),
            seed=int(row["seed"]),
        )
        for row in all_missing_rows
    ]
    aggregate["all_missing_by_n_subject"] = count_by_n_subject(all_missing_keys)
    with (audit_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    readme_text = build_readme(
        snapshot_root=snapshot_root,
        expected_parts=args.expected_parts,
        total_missing=len(all_missing_rows),
        expected_missing_total=args.expected_missing_total,
        part_summaries=part_summaries,
    )
    (audit_dir / "README.txt").write_text(readme_text, encoding="utf-8")

    sync_script = build_sync_script(
        audit_dir=audit_dir,
        snapshot_root=snapshot_root,
        remote_user_host=args.remote_user_host,
        remote_project_root=args.remote_project_root,
        remote_manifest_dir=args.remote_manifest_dir,
    )
    sync_path = audit_dir / "sync_backfill_inputs_to_hpc.sh"
    sync_path.write_text(sync_script, encoding="utf-8")
    sync_path.chmod(0o755)

    submit_script = build_submit_script(
        audit_dir=audit_dir,
        remote_project_root=args.remote_project_root,
        remote_manifest_dir=args.remote_manifest_dir,
        conda_module=args.conda_module,
        conda_env_path=args.conda_env_path,
        n_jobs=args.n_jobs,
        run_name_prefix=args.run_name_prefix,
    )
    submit_path = audit_dir / "submit_backfill.sh"
    submit_path.write_text(submit_script, encoding="utf-8")
    submit_path.chmod(0o755)

    if len(all_missing_rows) != args.expected_missing_total:
        raise SystemExit(
            f"audit mismatch: total missing {len(all_missing_rows)} != expected {args.expected_missing_total}"
        )

    missing_ns = {int(row["n_subject"]) for row in all_missing_rows}
    if missing_ns != {5000}:
        raise SystemExit(f"unexpected missing n_subject values: {sorted(missing_ns)}")

    duplicate_parts = [summary["part"] for summary in part_summaries if summary["duplicate_success_count"] > 0]
    invalid_parts = [summary["part"] for summary in part_summaries if summary["invalid_success_count"] > 0]
    if duplicate_parts:
        raise SystemExit(f"duplicate success keys found in parts: {duplicate_parts}")
    if invalid_parts:
        raise SystemExit(f"success rows outside expected key space found in parts: {invalid_parts}")

    print(f"Wrote audit outputs to {audit_dir}")
    print(f"Total missing tasks: {len(all_missing_rows)}")
    print(f"Generated sync script: {sync_path}")
    print(f"Generated submit script: {submit_path}")


if __name__ == "__main__":
    main()

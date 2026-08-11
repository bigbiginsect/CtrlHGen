"""Publish an audited Phase D parent without rewriting the original SFT best."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from akgr.reproduction.config import load_experiment_config


POINTER_STEM = "phase-d-parent"
SEMANTIC_METRICS = ("jaccard", "dice", "overlap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_validation(history_path: Path, stage_epoch: int) -> dict[str, Any]:
    matches = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if int(row["stage_epoch"]) == int(stage_epoch):
            matches.append(row)
    if len(matches) != 1 or matches[0].get("validation") is None:
        raise ValueError(
            f"Expected one validation record for epoch {stage_epoch} in {history_path}"
        )
    validation = dict(matches[0]["validation"])
    if validation.get("health_pass") is not True:
        raise ValueError("Phase D parent candidate did not pass validation health gates")
    return validation


def _verify_bakeoff_artifacts(summary: dict[str, Any]) -> None:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Bake-off summary does not contain artifact hashes")
    for raw_path, expected in artifacts.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"Bake-off artifact size mismatch: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"Bake-off artifact hash mismatch: {path}")


def _validate_bakeoff_decision(
    summary: dict[str, Any], *, candidate_label: str, baseline_label: str
) -> None:
    if summary.get("status") != "pass":
        raise ValueError("Bake-off summary did not pass its evaluation audit")
    for decode in ("greedy", "sampled"):
        report = summary.get("decoding", {}).get(decode)
        if not isinstance(report, dict):
            raise ValueError(f"Bake-off summary is missing {decode} results")
        candidate = report.get(candidate_label)
        delta = report.get(f"{candidate_label}_minus_{baseline_label}")
        if not isinstance(candidate, dict) or not isinstance(delta, dict):
            raise ValueError(
                f"Bake-off summary does not compare {candidate_label} against "
                f"{baseline_label} for {decode}"
            )
        if any(float(delta[name]) <= 0.0 for name in SEMANTIC_METRICS):
            raise ValueError(
                f"Phase D candidate must improve all semantic metrics for {decode}"
            )
        if float(delta["paper_average"]) <= 0.0:
            raise ValueError(f"Phase D candidate must improve paper_average for {decode}")
        if float(candidate["parse_ok"]) < 0.98 or float(candidate["eos_rate"]) < 0.98:
            raise ValueError(f"Phase D candidate failed {decode} parse/EOS gate")
        if float(candidate["condition_accuracy"]) < 0.90:
            raise ValueError(f"Phase D candidate failed {decode} condition gate")


def publish_phase_d_parent(
    *,
    checkpoint: Path,
    history_path: Path,
    bakeoff_summary_path: Path,
    expected_config_hash: str,
    expected_data_manifest_hash: str,
) -> Path:
    """Atomically publish ``phase-d-parent.json`` and its stable symlink."""
    checkpoint = checkpoint.expanduser().resolve()
    history_path = history_path.expanduser().resolve()
    bakeoff_summary_path = bakeoff_summary_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    metadata = _read_json(checkpoint / "metadata.json")
    if metadata.get("stage") != "conditional":
        raise ValueError("Phase D parent must be a conditional SFT checkpoint")
    if metadata.get("experiment_config_hash") != expected_config_hash:
        raise ValueError("Phase D parent config hash mismatch")
    if metadata.get("data_manifest_hash") != expected_data_manifest_hash:
        raise ValueError("Phase D parent data manifest hash mismatch")

    validation = _candidate_validation(history_path, int(metadata["stage_epoch"]))
    summary = _read_json(bakeoff_summary_path)
    if Path(summary["candidate_checkpoint"]).expanduser().resolve() != checkpoint:
        raise ValueError("Bake-off candidate checkpoint does not match requested parent")
    if summary.get("config_semantic_hash") != expected_config_hash:
        raise ValueError("Bake-off config hash mismatch")
    if summary.get("data_manifest_sha256") != expected_data_manifest_hash:
        raise ValueError("Bake-off data manifest hash mismatch")
    original_pointer_path = checkpoint.parent / "conditional-best.json"
    original = _read_json(original_pointer_path)
    if original.get("stage") not in (None, "conditional"):
        raise ValueError("Original training selection is not conditional")
    original_validation = original.get("validation", {})
    if original_validation and original_validation.get("health_pass") is not True:
        raise ValueError("Original conditional-best did not pass validation health gates")
    baseline_checkpoint = (checkpoint.parent / original["checkpoint"]).resolve()
    if Path(summary["baseline_checkpoint"]).expanduser().resolve() != baseline_checkpoint:
        raise ValueError("Bake-off baseline does not match original conditional-best")
    if baseline_checkpoint == checkpoint:
        raise ValueError("Phase D bake-off candidate and baseline must be different")
    baseline_epoch = original.get("stage_epoch")
    if baseline_epoch is None:
        baseline_epoch = original_validation.get("stage_epoch")
    if baseline_epoch is None:
        raise ValueError("Original conditional-best does not record stage_epoch")
    candidate_label = f"epoch{int(metadata['stage_epoch'])}"
    baseline_label = f"epoch{int(baseline_epoch)}"
    _validate_bakeoff_decision(
        summary, candidate_label=candidate_label, baseline_label=baseline_label
    )
    _verify_bakeoff_artifacts(summary)

    payload = {
        "schema_version": 1,
        "kind": "phase_d_parent",
        "stage": "conditional",
        "best_checkpoint": POINTER_STEM,
        "checkpoint": checkpoint.name,
        "original_training_selection": original["checkpoint"],
        "selection": {
            "policy": "post_phase_c_reproduction_oriented_bakeoff",
            "goal": "improve Jaccard, Dice, and Overlap while preserving healthy control and parsing",
            "validation_history": str(history_path),
            "bakeoff_summary": str(bakeoff_summary_path),
            "bakeoff_summary_sha256": _sha256(bakeoff_summary_path),
        },
        "validation": validation,
        "test_bakeoff": summary["decoding"],
    }
    pointer_path = checkpoint.parent / f"{POINTER_STEM}.json"
    link_path = checkpoint.parent / POINTER_STEM
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if pointer_path.exists() or link_path.exists() or link_path.is_symlink():
        if (
            pointer_path.is_file()
            and pointer_path.read_text(encoding="utf-8") == serialized
            and link_path.is_symlink()
            and link_path.resolve() == checkpoint
        ):
            return pointer_path
        raise FileExistsError(f"Phase D parent pointer already exists: {pointer_path}")

    temporary_pointer = pointer_path.with_name(f".{pointer_path.name}.tmp-{os.getpid()}")
    temporary_link = link_path.with_name(f".{link_path.name}.tmp-{os.getpid()}")
    try:
        temporary_pointer.write_text(serialized, encoding="utf-8")
        temporary_link.symlink_to(checkpoint.name, target_is_directory=True)
        os.replace(temporary_pointer, pointer_path)
        try:
            os.replace(temporary_link, link_path)
        except BaseException:
            pointer_path.unlink(missing_ok=True)
            raise
    finally:
        temporary_pointer.unlink(missing_ok=True)
        temporary_link.unlink(missing_ok=True)
    return pointer_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conditional-history", required=True)
    parser.add_argument("--bakeoff-summary", required=True)
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    pointer = publish_phase_d_parent(
        checkpoint=Path(args.checkpoint),
        history_path=Path(args.conditional_history),
        bakeoff_summary_path=Path(args.bakeoff_summary),
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_sha256(config.sampling_manifest_path),
    )
    print(pointer)


if __name__ == "__main__":
    main()

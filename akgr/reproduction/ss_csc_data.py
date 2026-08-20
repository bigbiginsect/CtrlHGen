"""Build frozen train-only solution-set pairs for SS-CSC Experiment A.

The command deliberately reads only the base training artifact.  Every query is
re-executed on the observable train graph, observation groups are split before
training, and both directions of every retained pair require a positive joint
relation-value branch marginal.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from akgr.abduction_model.experiment_runner import _graph_samplers
from akgr.dataloader import _load_manifest_artifact
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.sc_idc import (
    QueryExecutor,
    action_string,
    canonical_query_sha256,
    denotation_sha256,
    extract_semantic_slots,
    jaccard_score,
    neutralize_value_branches,
    parse_raw_query,
    pattern_signature,
    value_occurrences,
)
from akgr.reproduction.sc_idc_phase2_reward import (
    CachedQueryExecutor,
    StaticRelationMatcher,
    audit_relation_value,
)
from akgr.tokenizer import unique_condition_values_from_target
from akgr.utils.parsing_util import ans_shift_indices, list_to_str


SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return {"path": path.name, "count": count, "sha256": _sha256_file(path)}


def _load_pilot_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("method") != "SS-CSC":
        raise ValueError("Unsupported SS-CSC pilot config")
    if value.get("dataset") != "WN18RR" or value.get("condition") != "specific_relation":
        raise ValueError("SS-CSC pilot is frozen to WN18RR/specific_relation")
    return value


def _observation(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _observation_hash(values: Sequence[int]) -> str:
    return _sha256_bytes(_canonical_json(list(_observation(values))).encode("utf-8"))


def _topology(query, condition: int) -> str:
    occurrences = value_occurrences(query, "specific_relation", int(condition))
    if len(occurrences) > 1:
        return "repeated"
    return "first_only" if occurrences[0].ordinal == 0 else "non_first_only"


def _branch_audits(query, observation: frozenset[int], executor: QueryExecutor) -> dict[int, dict[str, Any]]:
    base = executor.execute(query)
    result = {}
    relation_values = sorted({slot.value for slot in extract_semantic_slots(query, "specific_relation")})
    for value in relation_values:
        neutral, branches = neutralize_value_branches(query, "specific_relation", value)
        neutral_denotation = executor.execute(neutral)
        delta = jaccard_score(base, observation) - jaccard_score(neutral_denotation, observation)
        result[int(value)] = {
            "condition_value": str(value),
            "branch_marginal_delta": float(delta),
            "branch_supported": bool(delta > 0.0),
            "occurrence_count": len(value_occurrences(query, "specific_relation", value)),
            "neutralized_branches": list(branches),
            "neutral_denotation_cardinality": len(neutral_denotation),
            "neutral_denotation_sha256": denotation_sha256(neutral_denotation),
        }
    return result


def _pair_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    deltas = (
        float(row["left"]["branch_marginal_delta"]),
        float(row["right"]["branch_marginal_delta"]),
    )
    return (-min(deltas), -sum(deltas), row["left"]["query_sha256"], row["right"]["query_sha256"])


def _choose_condition(exclusive: set[int], audits: Mapping[int, Mapping[str, Any]]):
    supported = [int(value) for value in exclusive if float(audits[int(value)]["branch_marginal_delta"]) > 0.0]
    if not supported:
        return None
    return sorted(supported, key=lambda value: (-float(audits[value]["branch_marginal_delta"]), value))[0]


def _direction(record: Mapping[str, Any], condition: int, audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": str(record["record_id"]),
        "query": list(record["query"]),
        "target": str(record["target"]),
        "query_sha256": str(record["query_sha256"]),
        "pattern_str": str(record["pattern_str"]),
        "pattern_signature": str(record["pattern_signature"]),
        "condition_value": str(condition),
        "topology": _topology(record["ast"], condition),
        **dict(audit),
    }


def _matched_diagnostic(
    side: dict[str, Any], *, observation: tuple[int, ...], matcher, cache, seed: int
) -> None:
    result = audit_relation_value(
        query=parse_raw_query(side["query"]),
        observation=observation,
        condition_value=int(side["condition_value"]),
        matcher=matcher,
        cache=cache,
        record_id=side["record_id"],
        reward_seed=int(seed),
    )
    side["matched_delta"] = result["matched_delta"]
    side["matched_classification"] = result["classification"]
    side["replacement_empty_rate"] = (
        sum(int(row["denotation_cardinality"]) == 0 for row in result["replacements"])
        / len(result["replacements"])
        if result["replacements"] else None
    )


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def build_audit(args: argparse.Namespace) -> Path:
    experiment = load_experiment_config(args.experiment_config)
    pilot_path = Path(args.pilot_config).expanduser().resolve()
    pilot = _load_pilot_config(pilot_path)
    if experiment.dataset != pilot["dataset"] or experiment.condition != pilot["condition"]:
        raise ValueError("Experiment config and SS-CSC pilot config disagree")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    sampling_path = experiment.sampling_manifest_path
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    raw_records = _load_manifest_artifact(str(sampling_path), sampling["artifacts"]["base"]["train"])
    graph_sampler = _graph_samplers(experiment)["train"]
    executor = QueryExecutor(graph_sampler)

    groups: defaultdict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    invalid_exact = []
    exact_records = []
    for raw in raw_records:
        observation = _observation(raw["answers"])
        ast = parse_raw_query(raw["query"])
        executed = executor.execute(ast)
        if executed != frozenset(observation):
            invalid_exact.append(str(raw["record_id"]))
            continue
        target = action_string(ast)
        record = {
            **raw,
            "ast": ast,
            "target": target,
            "query_sha256": canonical_query_sha256(ast),
            "pattern_signature": pattern_signature(ast),
            "relation_values": frozenset(
                int(value) for value in unique_condition_values_from_target("specific_relation", target)
            ),
        }
        groups[observation].append(record)
        exact_records.append(record)
    if invalid_exact:
        raise ValueError(f"Base train contains {len(invalid_exact)} non-exact records; first={invalid_exact[:3]}")

    distinct_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    duplicate_hypotheses = 0
    for observation, records in groups.items():
        by_hash = {}
        for record in sorted(records, key=lambda item: (item["query_sha256"], str(item["record_id"]))):
            if record["query_sha256"] in by_hash:
                duplicate_hypotheses += 1
                continue
            by_hash[record["query_sha256"]] = record
        distinct_groups[observation] = list(by_hash.values())

    eligible_group_count = 0
    candidate_pair_count = 0
    valid_pair_count = 0
    candidate_pairs_per_observation = []
    branch_supported_hypotheses = set()
    frozen_pairs: list[dict[str, Any]] = []
    branch_cache: dict[str, dict[int, dict[str, Any]]] = {}
    for observation, records in sorted(distinct_groups.items(), key=lambda item: _observation_hash(item[0])):
        candidates = []
        observation_candidates = 0
        for left, right in itertools.combinations(records, 2):
            left_only = set(left["relation_values"] - right["relation_values"])
            right_only = set(right["relation_values"] - left["relation_values"])
            if not left_only or not right_only:
                continue
            observation_candidates += 1
            candidate_pair_count += 1
            for record in (left, right):
                if record["query_sha256"] not in branch_cache:
                    branch_cache[record["query_sha256"]] = _branch_audits(
                        record["ast"], frozenset(observation), executor
                    )
            left_condition = _choose_condition(left_only, branch_cache[left["query_sha256"]])
            right_condition = _choose_condition(right_only, branch_cache[right["query_sha256"]])
            if left_condition is None or right_condition is None:
                continue
            branch_supported_hypotheses.update((left["query_sha256"], right["query_sha256"]))
            left_direction = _direction(
                left, left_condition, branch_cache[left["query_sha256"]][left_condition]
            )
            right_direction = _direction(
                right, right_condition, branch_cache[right["query_sha256"]][right_condition]
            )
            candidates.append({
                "observation": list(observation),
                "observation_sha256": _observation_hash(observation),
                "source": list_to_str(ans_shift_indices(list(observation))),
                "answer_cardinality": len(observation),
                "left": left_direction,
                "right": right_direction,
            })
        if observation_candidates:
            eligible_group_count += 1
            candidate_pairs_per_observation.append(observation_candidates)
        valid_pair_count += len(candidates)
        selected = sorted(candidates, key=_pair_rank)[: int(pilot["data"]["max_pairs_per_observation"])]
        for index, pair in enumerate(selected):
            pair["pair_id"] = f'{pair["observation_sha256"]}:{index}'
            frozen_pairs.append(pair)

    matcher = StaticRelationMatcher(graph_sampler)
    cache = CachedQueryExecutor(graph_sampler)
    for pair in frozen_pairs:
        for side_name in ("left", "right"):
            _matched_diagnostic(
                pair[side_name], observation=tuple(pair["observation"]), matcher=matcher,
                cache=cache, seed=int(pilot["seed"]),
            )

    modulus = int(pilot["data"]["validation_modulus"])
    remainder = int(pilot["data"]["validation_remainder"])
    train_pairs, validation_pairs = [], []
    for pair in frozen_pairs:
        bucket = int(pair["observation_sha256"], 16) % modulus
        (validation_pairs if bucket == remainder else train_pairs).append(pair)
    train_observations = {row["observation_sha256"] for row in train_pairs}
    validation_observations = {row["observation_sha256"] for row in validation_pairs}
    if train_observations & validation_observations:
        raise AssertionError("Pair train/validation observation overlap")
    minimum = int(pilot["data"]["min_train_pairs"])
    if len(train_pairs) < minimum:
        raise RuntimeError(f"Data gate failed: {len(train_pairs)} train pairs < {minimum}")

    # Existing-rule baseline: an update-matched deterministic sample of exact base
    # records, with a frozen uniform relation value selected independently of pairs.
    baseline_count = 2 * len(train_pairs)
    baseline_candidates = sorted(
        exact_records,
        key=lambda row: (_sha256_bytes(f'{pilot["seed"]}:{row["record_id"]}'.encode()), str(row["record_id"])),
    )
    if baseline_count > len(baseline_candidates):
        raise RuntimeError("Not enough exact base records for the update-matched baseline")
    baseline_rows = []
    for row in baseline_candidates[:baseline_count]:
        values = sorted(row["relation_values"])
        observation = _observation(row["answers"])
        baseline_rows.append({
            "example_id": str(row["record_id"]),
            "record_id": str(row["record_id"]),
            "observation": list(observation),
            "observation_sha256": _observation_hash(observation),
            "source": list_to_str(ans_shift_indices(list(observation))),
            "target": row["target"],
            "query_sha256": row["query_sha256"],
            "pattern_str": row["pattern_str"],
            "relation_values": [str(value) for value in values],
            "condition_policy": "uniform_unique_value_by_record_and_epoch",
        })

    artifacts = {
        "pair_train": _write_jsonl(output / "pair-train.jsonl", train_pairs),
        "pair_validation": _write_jsonl(output / "pair-validation.jsonl", validation_pairs),
        "single_target_train": _write_jsonl(output / "single-target-train.jsonl", baseline_rows),
    }
    pair_sides = [pair[side] for pair in frozen_pairs for side in ("left", "right")]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ss_csc_full_pair_data_audit",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
        },
        "inputs": {
            "experiment_config": str(experiment.source_path),
            "experiment_config_hash": experiment.semantic_hash,
            "pilot_config": str(pilot_path),
            "pilot_config_sha256": _sha256_file(pilot_path),
            "sampling_manifest": str(sampling_path),
            "sampling_manifest_sha256": _sha256_file(sampling_path),
            "base_train": sampling["artifacts"]["base"]["train"],
            "sealed_final_evaluation_loaded": False,
        },
        "data_gate": {
            "minimum_train_pairs": minimum,
            "actual_train_pairs": len(train_pairs),
            "passed": len(train_pairs) >= minimum,
            "train_validation_observation_overlap": 0,
        },
        "counts": {
            "raw_records": len(raw_records),
            "exact_records": len(exact_records),
            "unique_observations": len(groups),
            "distinct_hypotheses": sum(len(rows) for rows in distinct_groups.values()),
            "duplicate_hypotheses_removed": duplicate_hypotheses,
            "exclusive_relation_groups": eligible_group_count,
            "exclusive_candidate_pairs": candidate_pair_count,
            "bilateral_branch_supported_candidate_pairs": valid_pair_count,
            "frozen_pairs": len(frozen_pairs),
            "pair_train": len(train_pairs),
            "pair_validation": len(validation_pairs),
            "directional_train_examples": 2 * len(train_pairs),
            "single_target_train_examples": len(baseline_rows),
            "branch_supported_distinct_hypotheses": len(branch_supported_hypotheses),
        },
        "distributions": {
            "observation_group_cardinality": _counter_dict(len(rows) for rows in groups.values()),
            "distinct_hypotheses_per_observation": _counter_dict(len(rows) for rows in distinct_groups.values()),
            "candidate_pairs_per_eligible_observation": _counter_dict(candidate_pairs_per_observation),
            "frozen_pair_answer_cardinality": _counter_dict(pair["answer_cardinality"] for pair in frozen_pairs),
            "frozen_pair_left_pattern": _counter_dict(pair["left"]["pattern_str"] for pair in frozen_pairs),
            "frozen_pair_right_pattern": _counter_dict(pair["right"]["pattern_str"] for pair in frozen_pairs),
            "condition_topology": _counter_dict(side["topology"] for side in pair_sides),
        },
        "diagnostics": {
            "branch_supported_side_rate": 1.0,
            "bilateral_branch_supported_pair_rate": 1.0,
            "matched_selective_side_rate": sum(
                side.get("matched_delta") is not None and float(side["matched_delta"]) > 0.0
                for side in pair_sides
            ) / len(pair_sides),
            "replacement_empty_rate": sum(
                float(side["replacement_empty_rate"] or 0.0) for side in pair_sides
            ) / len(pair_sides),
            "graph_requests": cache.requests,
            "graph_executions": cache.executions,
        },
        "artifacts": artifacts,
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--pilot-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    path = build_audit(build_parser().parse_args(argv))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

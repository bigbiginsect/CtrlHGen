import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import networkx as nx

from akgr.kgdata import GraphSampler
from akgr.reproduction.sc_idc import (
    MatchedReplacementIndex,
    QueryExecutor,
    action_string,
    audit_slot,
    extract_semantic_slots,
    legacy_denotation,
    neutralize_slot_branch,
    parse_action,
    parse_raw_query,
    pattern_signature,
    replace_slot,
    select_semantic_slot,
    slot_counts,
)
from akgr.reproduction.sc_idc_audit import (
    load_verified_fresh_records,
    run_audit,
    run_self_check,
    stratified_records,
    summarize,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_pattern(pattern):
    encoded = pattern
    for operator in ("p", "e", "i", "u", "n"):
        encoded = encoded.replace(operator, f'"{operator}"')
    return json.loads(encoded.replace("(", "[").replace(")", "]"))


def _concrete_query(pattern, counters):
    operator, *children = pattern
    if operator == "e":
        entity = counters["entity"]
        counters["entity"] += 1
        return ["(", "e", "(", entity, ")", ")"]
    if operator == "p":
        relation = -counters["relation"]
        counters["relation"] += 1
        return [
            "(", "p", "(", relation, ")",
            *_concrete_query(children[0], counters), ")",
        ]
    result = ["(", operator]
    for child in children:
        result.extend(_concrete_query(child, counters))
    result.append(")")
    return result


def _audit_sampler():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(8))
    graph.add_edge(0, 2, key=0)
    graph.add_edge(1, 2, key=1)
    graph.add_edge(0, 3, key=2)
    graph.add_edge(4, 2, key=3)
    graph.add_edge(4, 3, key=3)
    graph.add_edge(5, 3, key=0)
    return GraphSampler(
        graph,
        {0: "+controlled", 1: "+dominant", 2: "+replacement", 3: "+broad"},
    )


class SCIDCASTTests(unittest.TestCase):
    def test_action_round_trip_and_slot_paths(self):
        query = parse_action("i -1 1 n -2 -3 2")
        self.assertEqual(action_string(query), "i -1 1 n -2 -3 2")
        self.assertEqual(pattern_signature(query), "i p e n p p e")
        self.assertEqual(slot_counts(query), {
            "specific_entity": 2,
            "specific_relation": 3,
        })
        relation_slots = extract_semantic_slots(query, "specific_relation")
        self.assertEqual([slot.path_string for slot in relation_slots], [
            "0:relation", "1/0:relation", "1/0/0:relation",
        ])

    def test_all_thirteen_patterns_match_legacy_executor(self):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(10))
        for source in range(10):
            for relation in range(3):
                graph.add_edge(source, (source + relation + 1) % 10, key=relation)
        sampler = GraphSampler(
            graph, {0: "+a", 1: "+b", 2: "+c"}
        )
        executor = QueryExecutor(sampler)
        with open("akgr/metadata/pattern_filtered.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 13)
        for row in rows:
            raw = _concrete_query(
                _parse_pattern(row["pattern_str"]),
                {"entity": 0, "relation": 0},
            )
            query = parse_raw_query(raw)
            self.assertEqual(
                executor.execute(query),
                legacy_denotation(sampler, query),
                row["pattern_abbr"],
            )

    def test_slot_selection_is_reproducible_and_reaches_duplicate_positions(self):
        query = parse_action("u -1 1 -1 1")
        first = select_semantic_slot(
            query, "specific_relation", record_id="r", epoch=9, seed=42
        )
        second = select_semantic_slot(
            query, "specific_relation", record_id="r", epoch=9, seed=42
        )
        self.assertEqual(first, second)
        observed = {
            select_semantic_slot(
                query, "specific_relation", record_id="r", epoch=epoch, seed=42
            ).ordinal
            for epoch in range(100)
        }
        self.assertEqual(observed, {0, 1})

    def test_replacement_preserves_pattern_and_counts(self):
        query = parse_action("i -1 1 -2 2")
        before = (pattern_signature(query), slot_counts(query))
        relation = extract_semantic_slots(query, "specific_relation")[1]
        entity = extract_semantic_slots(query, "specific_entity")[0]
        for replaced in (
            replace_slot(query, relation, -4),
            replace_slot(query, entity, 7),
        ):
            self.assertEqual((pattern_signature(replaced), slot_counts(replaced)), before)


class SCIDCCausalTests(unittest.TestCase):
    def setUp(self):
        self.sampler = _audit_sampler()
        self.matcher = MatchedReplacementIndex(self.sampler)

    def _audit_relation(self, action, observation, ordinal=1):
        query = parse_action(action)
        slot = extract_semantic_slots(query, "specific_relation")[ordinal]
        return audit_slot(
            query=query,
            observation=observation,
            slot=slot,
            matcher=self.matcher,
            record_id=action,
            seed=42,
            replacements=3,
        )

    def test_or_laundering_counterexample_is_rejected_despite_positive_match(self):
        row = self._audit_relation("u -2 2 -1 1", {2})
        self.assertAlmostEqual(row["branch_marginal_delta"], 0.0)
        self.assertGreater(row["matched_delta"], 0.0)
        self.assertEqual(row["classification"], "laundered")
        self.assertEqual(row["replacement_count"], 3)
        self.assertNotIn(
            int(row["condition_value"]),
            {candidate["token"] for candidate in row["replacements"]},
        )

    def test_useful_union_branch_has_positive_marginal(self):
        row = self._audit_relation("u -2 2 -3 1", {2, 3})
        self.assertGreater(row["branch_marginal_delta"], 0.0)

    def test_redundant_intersection_uses_universe_identity(self):
        row = self._audit_relation("i -2 2 -4 5", {2})
        self.assertAlmostEqual(row["branch_marginal_delta"], 0.0)
        self.assertEqual(row["neutralization"]["identity"], "universe")
        self.assertEqual(row["classification"], "laundered")

    def test_harmful_branch_has_negative_marginal(self):
        row = self._audit_relation("u -2 2 -3 1", {2})
        self.assertLess(row["branch_marginal_delta"], 0.0)

    def test_single_chain_uses_empty_root_baseline(self):
        query = parse_action("-2 2")
        slot = extract_semantic_slots(query, "specific_relation")[0]
        neutral = neutralize_slot_branch(query, slot)
        row = audit_slot(
            query=query,
            observation={2},
            slot=slot,
            matcher=self.matcher,
            record_id="root",
            seed=42,
        )
        self.assertEqual(neutral.identity, "empty")
        self.assertIsNone(neutral.branch_operator)
        self.assertAlmostEqual(row["branch_marginal_delta"], 1.0)

    def test_negation_matches_legacy_semantics(self):
        query = parse_action("n -1 1")
        self.assertEqual(
            QueryExecutor(self.sampler).execute(query),
            legacy_denotation(self.sampler, query),
        )

    def test_entity_matching_is_deterministic_and_structure_preserving(self):
        query = parse_action("-1 1")
        slot = extract_semantic_slots(query, "specific_entity")[0]
        kwargs = dict(
            query=query,
            slot=slot,
            observation={2},
            replacements=3,
            record_id="entity",
            seed=42,
        )
        first = self.matcher.matches(**kwargs)
        second = self.matcher.matches(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotIn(slot.value, {candidate.token for candidate in first})
        for candidate in first:
            replaced = replace_slot(query, slot, candidate.token)
            self.assertEqual(pattern_signature(replaced), pattern_signature(query))
            self.assertEqual(slot_counts(replaced), slot_counts(query))

    def test_standalone_self_check(self):
        self.assertEqual(run_self_check()["status"], "pass")


class SCIDCAuditContractTests(unittest.TestCase):
    def test_end_to_end_audit_artifacts_are_deterministic(self):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(12))
        for source in range(12):
            for relation in range(4):
                graph.add_edge(source, (source + relation + 1) % 12, key=relation)
        sampler = GraphSampler(
            graph,
            {relation: f"+r{relation}" for relation in range(4)},
        )
        executor = QueryExecutor(sampler)

        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            sampling_manifest = directory / "sampling-manifest.json"
            sampling_manifest.write_text("{}\n", encoding="utf-8")
            fresh_data = directory / "fresh.jsonl"
            records = []
            with open(
                "akgr/metadata/pattern_table.csv", newline="", encoding="utf-8"
            ) as handle:
                patterns = [
                    row for row in csv.DictReader(handle)
                    if int(row["original_depth"]) <= 2
                ]
            self.assertEqual(len(patterns), 13)
            for row in patterns:
                raw_query = _concrete_query(
                    _parse_pattern(row["original"]),
                    {"entity": 0, "relation": 0},
                )
                answers = sorted(executor.execute(parse_raw_query(raw_query)))
                records.append({
                    "answers": answers,
                    "query": raw_query,
                    "pattern_str": row["original"],
                    "record_id": f"reference-{row['pattern_abbr']}",
                })
            fresh_data.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            fresh_manifest = directory / "fresh-manifest.json"
            fresh_manifest.write_text(json.dumps({
                "schema_version": 1,
                "kind": "phase_d_repaired_pilot_rl_only",
                "config_semantic_hash": "config-hash",
                "original_sampling_manifest_sha256": _sha256(sampling_manifest),
                "artifact": {
                    "path": fresh_data.name,
                    "sha256": _sha256(fresh_data),
                    "count": len(records),
                },
            }), encoding="utf-8")
            experiment_config = directory / "experiment.yml"
            experiment_config.write_text("schema_version: 1\n", encoding="utf-8")
            config = SimpleNamespace(
                source_path=experiment_config,
                dataset="WN18RR",
                condition="pattern",
                semantic_hash="config-hash",
                data_hash="data-hash",
                kg_hash="kg-hash",
                sampling_manifest_path=sampling_manifest,
            )

            output_paths = [directory / "audit-a", directory / "audit-b"]
            with mock.patch(
                "akgr.reproduction.sc_idc_audit.load_experiment_config",
                return_value=config,
            ), mock.patch(
                "akgr.reproduction.sc_idc_audit._graph_samplers",
                return_value={"train": sampler},
            ):
                for output_path in output_paths:
                    run_audit(SimpleNamespace(
                        split="train",
                        output_dir=output_path,
                        experiment_config=config.source_path,
                        fresh_manifest=fresh_manifest,
                        per_pattern=1,
                        seed=42,
                        condition_kind="both",
                        replacements=3,
                    ))

            for artifact in ("slot-audit.jsonl", "summary.json"):
                self.assertEqual(
                    (output_paths[0] / artifact).read_bytes(),
                    (output_paths[1] / artifact).read_bytes(),
                    artifact,
                )
            summary = json.loads(
                (output_paths[0] / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["selected_record_count"], 13)
            self.assertTrue(summary["slot_weight_contract"]["pass"])
            self.assertEqual(summary["overall"]["reference_exact_rate"], 1.0)
            self.assertEqual(summary["overall"]["reference_jaccard_one_rate"], 1.0)
            self.assertEqual(
                summary["overall"]["legacy_executor_parity_rate"], 1.0
            )
            self.assertEqual(summary["overall"]["structure_preservation_rate"], 1.0)
            self.assertEqual(summary["overall"]["unscorable_rate"], 0.0)
            self.assertTrue(summary["integrity_gates"]["all_single_run_gates_pass"])

    def test_fresh_manifest_hash_and_count_are_enforced(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            data_path = directory / "fresh.jsonl"
            row = {
                "answers": [1],
                "query": ["(", "p", "(", 0, ")", "(", "e", "(", 0, ")", ")", ")"],
                "pattern_str": "(p,(e))",
                "record_id": "rec-1",
            }
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_path = directory / "fresh-manifest.json"
            manifest = {
                "schema_version": 1,
                "kind": "phase_d_repaired_pilot_rl_only",
                "config_semantic_hash": "config",
                "original_sampling_manifest_sha256": "sampling",
                "artifact": {
                    "path": data_path.name,
                    "sha256": _sha256(data_path),
                    "count": 1,
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            rows, _, resolved = load_verified_fresh_records(
                manifest_path,
                expected_config_hash="config",
                expected_sampling_manifest_hash="sampling",
            )
            self.assertEqual(rows[0]["record_id"], "rec-1")
            self.assertEqual(resolved, data_path.resolve())

            with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                load_verified_fresh_records(
                    manifest_path,
                    expected_config_hash="tampered-config",
                    expected_sampling_manifest_hash="sampling",
                )
            with self.assertRaisesRegex(ValueError, "sampling-manifest hash mismatch"):
                load_verified_fresh_records(
                    manifest_path,
                    expected_config_hash="config",
                    expected_sampling_manifest_hash="tampered-sampling",
                )

            data_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_verified_fresh_records(
                    manifest_path,
                    expected_config_hash="config",
                    expected_sampling_manifest_hash="sampling",
                )

    def test_stratified_selection_is_deterministic_and_balanced(self):
        with open("akgr/metadata/pattern_table.csv", newline="", encoding="utf-8") as handle:
            patterns = [
                row for row in csv.DictReader(handle)
                if int(row["original_depth"]) <= 2
            ]
        records = []
        for row in patterns:
            for ordinal in range(2):
                records.append({
                    "answers": [0],
                    "query": [],
                    "pattern_str": row["original"],
                    "record_id": f"{row['pattern_abbr']}-{ordinal}",
                })
        first = stratified_records(records, per_pattern=1, seed=42)
        second = stratified_records(records, per_pattern=1, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 13)
        self.assertEqual(len({row["pattern_abbreviation"] for row in first}), 13)

    def test_summary_checks_per_record_kind_weights(self):
        sampler = _audit_sampler()
        matcher = MatchedReplacementIndex(sampler)
        query = parse_action("u -2 2 -1 1")
        slots = extract_semantic_slots(query, "specific_relation")
        rows = []
        for slot in slots:
            row = audit_slot(
                query=query,
                observation={2},
                slot=slot,
                matcher=matcher,
                record_id="weight",
                seed=42,
            )
            row.update({
                "slot_weight": 0.5,
                "pattern_abbreviation": "2u",
                "legacy_executor_parity": True,
            })
            rows.append(row)
        result = summarize(rows, selected_record_count=1)
        self.assertTrue(result["slot_weight_contract"]["pass"])
        self.assertAlmostEqual(
            result["slot_weight_contract"]["max_absolute_error"], 0.0
        )

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            args = SimpleNamespace(
                split="train",
                output_dir=raw_dir,
                condition_kind="both",
                per_pattern=16,
                replacements=3,
                seed=42,
            )
            with self.assertRaises(FileExistsError):
                run_audit(args)


if __name__ == "__main__":
    unittest.main()

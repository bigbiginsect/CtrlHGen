import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import networkx as nx

from akgr.reproduction.sc_idc import QueryExecutor, parse_action
from akgr.reproduction.ss_csc_common import load_pair_artifact, summarize_pair_likelihood
from akgr.reproduction.ss_csc_data import _branch_audits, _choose_condition, _pair_rank
from akgr.reproduction.ss_csc_evaluate import _bootstrap_delta


class _Sampler:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(range(5))
        self.graph.add_edge(0, 2, key=0)
        self.graph.add_edge(0, 3, key=0)
        self.graph.add_edge(1, 2, key=1)
        self.graph.add_edge(1, 4, key=1)


class SSCscDataTests(unittest.TestCase):
    def test_branch_audit_and_condition_selection(self):
        executor = QueryExecutor(_Sampler())
        query = parse_action("i -1 1 -2 2")
        observation = frozenset({2})
        audits = _branch_audits(query, observation, executor)
        self.assertEqual(set(audits), {-1, -2})
        self.assertGreater(audits[-1]["branch_marginal_delta"], 0.0)
        self.assertGreater(audits[-2]["branch_marginal_delta"], 0.0)
        self.assertEqual(_choose_condition({-1, -2}, audits), -2)

    def test_pair_rank_prefers_larger_minimum_margin(self):
        weak = {
            "left": {"branch_marginal_delta": 0.1, "query_sha256": "a"},
            "right": {"branch_marginal_delta": 0.2, "query_sha256": "b"},
        }
        strong = {
            "left": {"branch_marginal_delta": 0.3, "query_sha256": "c"},
            "right": {"branch_marginal_delta": 0.3, "query_sha256": "d"},
        }
        self.assertEqual(sorted([weak, strong], key=_pair_rank)[0], strong)

    def test_pair_artifact_fails_closed_on_hash_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = root / "pair-validation.jsonl"
            rows.write_text('{"pair_id":"p"}\n', encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(rows.read_bytes()).hexdigest()
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "kind": "ss_csc_full_pair_data_audit", "status": "passed",
                "data_gate": {"passed": True},
                "artifacts": {"pair_validation": {
                    "path": rows.name, "count": 1, "sha256": digest,
                }},
            }), encoding="utf-8")
            _, loaded, _ = load_pair_artifact(summary, "pair_validation")
            self.assertEqual(loaded[0]["pair_id"], "p")
            rows.write_text('{"pair_id":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_pair_artifact(summary, "pair_validation")


class SSCscEvaluationTests(unittest.TestCase):
    def test_likelihood_summary_and_paired_bootstrap(self):
        rows = [{
            "h1_c1_mean_logp": -1.0, "h2_c2_mean_logp": -1.2,
            "h1_c2_swapped_mean_logp": -1.5, "h2_c1_swapped_mean_logp": -1.6,
            "h1_prompt_margin": 0.5, "h2_prompt_margin": 0.4,
            "h1_margin_satisfied": True, "h2_margin_satisfied": True,
            "symmetric_margin_satisfied": True,
            "condition_consistent_selection_accuracy": 1.0,
            "bilateral_condition_consistent_selection": True,
        }]
        summary = summarize_pair_likelihood(rows)
        self.assertAlmostEqual(summary["assigned_target_mean_logp"], -1.1)
        self.assertEqual(summary["symmetric_margin_satisfaction_rate"], 1.0)
        bootstrap = _bootstrap_delta([1.0] * 10, [0.0] * 10, samples=100, seed=42)
        self.assertEqual(bootstrap["delta"], 1.0)
        self.assertEqual(bootstrap["ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

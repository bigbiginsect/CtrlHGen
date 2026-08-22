import unittest

import networkx as nx

from akgr.kgdata.kgclass import GraphSampler
from akgr.reproduction.specific_relation_posthoc_audits import (
    _qualifies,
    build_valid_exclusive_sampler,
    select_with,
)


def _candidate(
    *,
    exact=False,
    nominal=False,
    branch=False,
    semantic=0.0,
    logp=-1.0,
    sha="a",
):
    return {
        "exact": exact,
        "nominal": nominal,
        "branch_supported": branch,
        "semantic_average": semantic,
        "model_mean_log_probability": logp,
        "canonical_hypothesis_sha256": sha,
    }


class SelectorAblationTests(unittest.TestCase):
    def test_exact_semantic_is_condition_blind(self):
        candidates = [
            _candidate(exact=True, logp=-1.0, sha="b"),
            _candidate(exact=True, nominal=True, branch=True, logp=-2.0, sha="a"),
        ]
        self.assertEqual(select_with("exact_semantic", candidates), (0, "exact"))
        self.assertEqual(
            select_with("branch_aware", candidates),
            (1, "exact+branch_supported"),
        )

    def test_nominal_ablation_ignores_branch_support(self):
        candidates = [
            _candidate(exact=True, nominal=True, logp=-1.0, sha="b"),
            _candidate(
                exact=True,
                nominal=True,
                branch=True,
                logp=-2.0,
                sha="a",
            ),
        ]
        self.assertEqual(
            select_with("nominal_aware", candidates),
            (0, "exact+nominal"),
        )
        self.assertEqual(
            select_with("branch_aware", candidates),
            (1, "exact+branch_supported"),
        )

    def test_exact_branch_omits_nominal_fallback(self):
        candidates = [
            _candidate(exact=True, nominal=True, logp=-2.0, sha="a"),
            _candidate(exact=True, logp=-1.0, sha="b"),
        ]
        self.assertEqual(select_with("exact_branch", candidates), (1, "exact"))
        self.assertEqual(
            select_with("nominal_aware", candidates),
            (0, "exact+nominal"),
        )

    def test_single_signal_selectors(self):
        candidates = [
            _candidate(semantic=0.9, logp=-4.0, sha="a"),
            _candidate(semantic=0.5, logp=-0.1, sha="b"),
        ]
        self.assertEqual(select_with("semantic_only", candidates)[0], 0)
        self.assertEqual(select_with("likelihood_only", candidates)[0], 1)
        self.assertEqual(select_with("first_sample", candidates)[0], 0)

    def test_joint_availability_requires_exact_and_control(self):
        candidate = _candidate(
            exact=False,
            nominal=True,
            branch=True,
        )
        candidate.update({
            "parse_ok": True,
            "nonroot_branch_supported": True,
        })
        self.assertTrue(_qualifies(candidate, "branch_supported"))
        self.assertFalse(_qualifies(candidate, "exact_branch_supported"))
        candidate["exact"] = True
        self.assertTrue(_qualifies(candidate, "exact_nominal"))
        self.assertTrue(_qualifies(candidate, "exact_branch_supported"))
        self.assertTrue(_qualifies(candidate, "exact_nonroot_branch_supported"))


class GraphViewTests(unittest.TestCase):
    @staticmethod
    def _sampler(edges, nodes=(0, 1, 2, 3)):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(nodes)
        for source, target, relation in edges:
            graph.add_edge(source, target, key=relation)
        return GraphSampler(graph, {0: "+r", 1: "-r"})

    def test_valid_exclusive_is_disjoint_and_preserves_universe(self):
        train = self._sampler([(0, 1, 0), (1, 0, 1)])
        valid = self._sampler([
            (0, 1, 0),
            (1, 0, 1),
            (2, 3, 0),
            (3, 2, 1),
        ])
        exclusive, contract = build_valid_exclusive_sampler(train, valid)
        self.assertEqual(
            set(exclusive.graph.edges(keys=True)),
            {(2, 3, 0), (3, 2, 1)},
        )
        self.assertEqual(set(exclusive.graph.nodes), set(valid.graph.nodes))
        self.assertTrue(all(contract["checks"].values()))


if __name__ == "__main__":
    unittest.main()

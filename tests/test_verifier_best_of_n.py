import unittest

from akgr.reproduction.verifier_best_of_n import p1_gate, select_candidate


def _candidate(*, exact=False, nominal=False, branch=False, semantic=0.0, logp=-1.0, sha="a"):
    return {
        "exact": exact,
        "nominal": nominal,
        "branch_supported": branch,
        "semantic_average": semantic,
        "model_mean_log_probability": logp,
        "canonical_hypothesis_sha256": sha,
    }


class VerifierSelectionTests(unittest.TestCase):
    def test_exact_branch_precedes_higher_semantic_nonexact(self):
        candidates = [
            _candidate(semantic=0.999, logp=-0.1, sha="b"),
            _candidate(exact=True, nominal=True, branch=True, semantic=0.9, logp=-5.0, sha="a"),
        ]
        self.assertEqual(select_candidate(candidates), (1, "exact+branch_supported"))

    def test_likelihood_then_hash_break_ties(self):
        candidates = [
            _candidate(exact=True, logp=-2.0, sha="a"),
            _candidate(exact=True, logp=-1.0, sha="z"),
            _candidate(exact=True, logp=-1.0, sha="b"),
        ]
        self.assertEqual(select_candidate(candidates), (2, "exact"))

    def test_semantic_used_only_without_exact_candidate(self):
        candidates = [
            _candidate(nominal=True, branch=True, semantic=0.5, logp=-0.1, sha="a"),
            _candidate(semantic=0.8, logp=-4.0, sha="b"),
        ]
        self.assertEqual(select_candidate(candidates), (1, "highest_semantic_average"))


class VerifierGateTests(unittest.TestCase):
    @staticmethod
    def _summary(semantic, branch, parse, eos, switch=None):
        result = {
            "semantic_average": semantic,
            "branch_supported_rate": branch,
            "parse_rate": parse,
            "eos_rate": eos,
            "cost": {"wall_seconds": 1.0, "graph_executions": 1},
        }
        if switch is not None:
            result["pair_switch"] = {"bilateral_exact_ast_switch_rate": switch}
        return result

    def test_gate_passes_at_frozen_thresholds(self):
        original = {
            "greedy": self._summary(0.7, 0.5, 1.0, 1.0),
            "sample_k4": self._summary(0.8, 0.6, 0.995, 0.995),
        }
        pair = {
            "greedy": self._summary(0.9, 0.5, 1.0, 1.0, 0.2),
            "sample_k4": self._summary(0.99, 0.7, 0.995, 0.995, 0.65766),
        }
        self.assertTrue(p1_gate(original, pair)["passed"])

    def test_gate_fails_semantic_regression(self):
        original = {
            "greedy": self._summary(0.7, 0.5, 1.0, 1.0),
            "sample_k4": self._summary(0.69, 0.6, 1.0, 1.0),
        }
        pair = {
            "greedy": self._summary(0.9, 0.5, 1.0, 1.0, 0.2),
            "sample_k4": self._summary(0.99, 0.7, 1.0, 1.0, 0.8),
        }
        gate = p1_gate(original, pair)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["original_semantic_non_decrease"])


if __name__ == "__main__":
    unittest.main()

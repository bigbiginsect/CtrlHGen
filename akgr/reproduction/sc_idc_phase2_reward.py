"""Value-level SC-IDC reward core for the Phase 2 specific-relation study."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import random
import time
from typing import Any, Iterable

from akgr.reproduction.sc_idc import (
    QueryExecutor,
    QueryNode,
    canonical_query_sha256,
    denotation_sha256,
    jaccard_score,
    neutralize_value_branches,
    pattern_signature,
    replace_value,
    slot_counts,
    value_occurrences,
)


SCHEMA_VERSION = 2
CLASSIFICATIONS = (
    "branch_nonmarginal",
    "branch_supported_nonselective",
    "branch_supported_selective",
    "unscorable",
)


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _log_bin(value: int) -> int:
    return int(math.floor(math.log2(max(0, int(value)) + 1)))


def _counter_cosine(left: Counter[Any], right: Counter[Any]) -> float:
    common = set(left) & set(right)
    numerator = math.fsum(float(left[key]) * float(right[key]) for key in common)
    left_norm = math.sqrt(math.fsum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(math.fsum(float(value) ** 2 for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def set_semantic_scores(
    prediction: Iterable[int], observation: Iterable[int]
) -> dict[str, float]:
    predicted = frozenset(int(value) for value in prediction)
    observed = frozenset(int(value) for value in observation)
    intersection = len(predicted & observed)
    union = len(predicted | observed)
    jaccard = 1.0 if union == 0 else intersection / union
    denominator = len(predicted) + len(observed)
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    minimum = min(len(predicted), len(observed))
    overlap = 1.0 if minimum == 0 and not predicted and not observed else intersection / (minimum + 1e-5)
    return {"jaccard": jaccard, "dice": dice, "overlap": overlap}


class CachedQueryExecutor:
    """Canonical-AST cache with explicit execution accounting."""

    def __init__(self, graph_sampler) -> None:
        self.executor = QueryExecutor(graph_sampler)
        self.cache: dict[str, frozenset[int]] = {}
        self.requests = 0
        self.executions = 0

    def execute(self, query: QueryNode) -> tuple[frozenset[int], bool]:
        self.requests += 1
        # Neutralized internal queries may contain identity leaves and therefore
        # cannot be serialized as model actions. QueryNode repr is canonical.
        key = repr(query)
        if key in self.cache:
            return self.cache[key], True
        result = self.executor.execute(query)
        self.cache[key] = result
        self.executions += 1
        return result, False


class StaticRelationMatcher:
    """Compute-bounded relation matcher using only cached train-graph features."""

    def __init__(self, graph_sampler) -> None:
        self.graph_sampler = graph_sampler
        graph = graph_sampler.graph
        relation_ids = {int(key) for _, _, key in graph.edges(keys=True)}
        self.relation_ids = tuple(sorted(relation_ids))
        self.edge_counts = Counter(int(key) for _, _, key in graph.edges(keys=True))
        incidence: dict[int, Counter[tuple[str, int]]] = {}
        for node in graph.nodes:
            profile: Counter[tuple[str, int]] = Counter()
            for _, _, key in graph.out_edges(node, keys=True):
                profile[("out", int(key))] += 1
            for _, _, key in graph.in_edges(node, keys=True):
                profile[("in", int(key))] += 1
            incidence[int(node)] = profile
        source_nodes: defaultdict[int, set[int]] = defaultdict(set)
        target_nodes: defaultdict[int, set[int]] = defaultdict(set)
        for source, target, key in graph.edges(keys=True):
            source_nodes[int(key)].add(int(source))
            target_nodes[int(key)].add(int(target))
        self.head_profiles: dict[int, Counter[Any]] = {}
        self.tail_profiles: dict[int, Counter[Any]] = {}
        for relation_id in self.relation_ids:
            head: Counter[Any] = Counter()
            tail: Counter[Any] = Counter()
            for node in source_nodes[relation_id]:
                head.update(incidence[node])
            for node in target_nodes[relation_id]:
                tail.update(incidence[node])
            self.head_profiles[relation_id] = head
            self.tail_profiles[relation_id] = tail
        self._rank_cache: dict[int, tuple[dict[str, Any], ...]] = {}

    def _direction(self, relation_id: int) -> str:
        name = str(getattr(self.graph_sampler, "id2rel", {}).get(int(relation_id), ""))
        if name.startswith("+"):
            return "forward"
        if name.startswith("-"):
            return "reverse"
        return "unknown"

    def ranked(self, condition_value: int) -> tuple[dict[str, Any], ...]:
        original_id = abs(int(condition_value)) - 1
        if original_id not in self.relation_ids:
            return ()
        if original_id in self._rank_cache:
            return self._rank_cache[original_id]
        original_direction = self._direction(original_id)
        original_frequency_bin = _log_bin(self.edge_counts[original_id])
        rows = []
        for relation_id in self.relation_ids:
            if relation_id == original_id:
                continue
            head = _counter_cosine(
                self.head_profiles[original_id], self.head_profiles[relation_id]
            )
            tail = _counter_cosine(
                self.tail_profiles[original_id], self.tail_profiles[relation_id]
            )
            endpoint = (head + tail) / 2.0
            direction_level = int(self._direction(relation_id) != original_direction)
            frequency_distance = abs(
                _log_bin(self.edge_counts[relation_id]) - original_frequency_bin
            )
            rows.append({
                "token": -(relation_id + 1),
                "relation_id": relation_id,
                "fallback": "same_direction" if direction_level == 0 else "all_relations",
                "direction_fallback_level": direction_level,
                "edge_frequency_log_bin": _log_bin(self.edge_counts[relation_id]),
                "edge_frequency_bin_distance": frequency_distance,
                "head_endpoint_profile_cosine": head,
                "tail_endpoint_profile_cosine": tail,
                "endpoint_profile_cosine": endpoint,
            })
        rows.sort(key=lambda row: (
            row["direction_fallback_level"],
            row["edge_frequency_bin_distance"],
            -row["endpoint_profile_cosine"],
            row["relation_id"],
        ))
        self._rank_cache[original_id] = tuple(rows)
        return self._rank_cache[original_id]

    def select(
        self,
        *,
        condition_value: int,
        query: QueryNode,
        record_id: str,
        reward_seed: int,
        replacements: int = 3,
    ) -> tuple[dict[str, Any], ...]:
        if int(replacements) != 3:
            raise ValueError("Phase 2 training matcher is frozen to K=3")
        ranked = self.ranked(condition_value)
        if len(ranked) < int(replacements):
            return ()
        best = list(ranked[:8])
        seed = _stable_seed(
            int(reward_seed), str(record_id), canonical_query_sha256(query), int(condition_value)
        )
        indices = sorted(random.Random(seed).sample(range(len(best)), int(replacements)))
        return tuple({**best[index], "static_rank": index} for index in indices)


def audit_relation_value(
    *,
    query: QueryNode,
    observation: Iterable[int],
    condition_value: int,
    matcher: StaticRelationMatcher,
    cache: CachedQueryExecutor,
    record_id: str,
    reward_seed: int,
    epsilon: float = 0.0,
    tau_marg: float = 0.1,
    tau_match: float = 0.1,
    base_denotation: frozenset[int] | None = None,
) -> dict[str, Any]:
    if tau_marg <= 0.0 or tau_match <= 0.0:
        raise ValueError("SC-IDC normalization taus must be positive")
    started = time.perf_counter()
    before_executions = cache.executions
    before_requests = cache.requests
    occurrences = value_occurrences(query, "specific_relation", int(condition_value))
    if not occurrences:
        raise ValueError("Value audit requires nominal adherence")
    if base_denotation is None:
        base, base_cache_hit = cache.execute(query)
    else:
        base, base_cache_hit = base_denotation, True
    observation_set = frozenset(int(value) for value in observation)
    base_score = jaccard_score(base, observation_set)
    neutral_query, neutralized_branches = neutralize_value_branches(
        query, "specific_relation", int(condition_value)
    )
    neutral, neutral_cache_hit = cache.execute(neutral_query)
    neutral_score = jaccard_score(neutral, observation_set)
    branch_delta = base_score - neutral_score
    selected = matcher.select(
        condition_value=int(condition_value), query=query, record_id=str(record_id),
        reward_seed=int(reward_seed), replacements=3,
    )
    original_pattern = pattern_signature(query)
    original_counts = slot_counts(query)
    replacement_rows = []
    for candidate in selected:
        candidate_query = replace_value(
            query, "specific_relation", int(condition_value), int(candidate["token"])
        )
        denotation, cache_hit = cache.execute(candidate_query)
        score = jaccard_score(denotation, observation_set)
        replacement_rows.append({
            **candidate,
            "query_sha256": canonical_query_sha256(candidate_query),
            "semantic_score": score,
            "delta": base_score - score,
            "denotation_cardinality": len(denotation),
            "denotation_sha256": denotation_sha256(denotation),
            "cache_hit": cache_hit,
            "structure_preserved": (
                pattern_signature(candidate_query) == original_pattern
                and slot_counts(candidate_query) == original_counts
                and len(value_occurrences(
                    candidate_query, "specific_relation", int(candidate["token"])
                )) >= len(occurrences)
            ),
        })
    matched_delta = (
        sum(row["delta"] for row in replacement_rows) / len(replacement_rows)
        if len(replacement_rows) == 3 else None
    )
    if matched_delta is None:
        classification = "unscorable"
    elif branch_delta <= float(epsilon):
        classification = "branch_nonmarginal"
    elif matched_delta <= float(epsilon):
        classification = "branch_supported_nonselective"
    else:
        classification = "branch_supported_selective"
    normalized_branch = max(0.0, min(1.0, (branch_delta - epsilon) / tau_marg))
    normalized_match = (
        0.0 if matched_delta is None
        else max(0.0, min(1.0, (matched_delta - epsilon) / tau_match))
    )
    raw_bss = base_score * normalized_branch * normalized_match
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(record_id),
        "condition_kind": "specific_relation",
        "condition_value": str(condition_value),
        "occurrence_count": len(occurrences),
        "occurrence_paths": [slot.path_string for slot in occurrences],
        "nominal_adherence": True,
        "query_sha256": canonical_query_sha256(query),
        "base_semantic_score": base_score,
        "base_denotation_cardinality": len(base),
        "base_denotation_sha256": denotation_sha256(base),
        "base_cache_hit": base_cache_hit,
        "neutralization": {
            "branches": list(neutralized_branches),
            "semantic_score": neutral_score,
            "denotation_cardinality": len(neutral),
            "denotation_sha256": denotation_sha256(neutral),
            "cache_hit": neutral_cache_hit,
        },
        "branch_marginal_delta": branch_delta,
        "matched_delta": matched_delta,
        "replacements": replacement_rows,
        "replacement_count": len(replacement_rows),
        "classification": classification,
        "raw_bss": raw_bss,
        "execution_requests": cache.requests - before_requests,
        "graph_executions": cache.executions - before_executions,
        "wall_time_seconds": time.perf_counter() - started,
        "terminology": {
            "branch_supported_selectivity_only": True,
            "predicate_necessity_claim": False,
            "natural_branch_nonmarginal_called_laundering": False,
        },
    }

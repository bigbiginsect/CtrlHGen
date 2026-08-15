"""Offline SC-IDC primitives for semantic-control falsification audits.

This module is deliberately independent of the model and training stack.  It
operates on CtrlHGen's prefix action language, executes queries against an
already loaded :class:`~akgr.kgdata.kgclass.GraphSampler`, and exposes the two
quantities needed by the first SC-IDC audit:

* a nearest-branch marginal contribution obtained with logical identity
  elements; and
* a matched value-replacement advantage for one explicit entity/relation slot.

Matched replacement alone is not a test of non-redundancy.  A redundant union
branch can contain the requested value while a replacement introduces false
positives, making the matched delta positive.  ``audit_slot`` therefore keeps
the two measurements separate and only labels a slot ``strict_effective`` when
both are positive.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Literal, Mapping, Sequence

from akgr.utils.parsing_util import (
    list_to_str,
    qry_shift_indices,
    qry_str_2_actionstr,
)


SlotKind = Literal["specific_entity", "specific_relation"]
SET_OPERATORS = {"i", "u"}
QUERY_OPERATORS = {"e", "p", "i", "u", "n", "empty", "universe"}


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _log_bin(value: int) -> int:
    return int(math.floor(math.log2(max(0, int(value)) + 1)))


def _jaccard_sets(left: set[Any] | frozenset[Any], right: set[Any] | frozenset[Any]) -> float:
    union = set(left) | set(right)
    if not union:
        return 1.0
    return len(set(left) & set(right)) / len(union)


def jaccard_score(prediction: Iterable[int], observation: Iterable[int]) -> float:
    """Return deterministic set Jaccard, defining two empty sets as equal."""
    return _jaccard_sets(frozenset(prediction), frozenset(observation))


def denotation_sha256(values: Iterable[int]) -> str:
    payload = json.dumps(sorted({int(value) for value in values}), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryNode:
    """Immutable AST for the binary CtrlHGen query language."""

    operator: str
    value: int | None = None
    children: tuple["QueryNode", ...] = ()

    def __post_init__(self) -> None:
        if self.operator not in QUERY_OPERATORS:
            raise ValueError(f"Unsupported query operator: {self.operator!r}")
        expected_children = {
            "e": 0,
            "p": 1,
            "i": 2,
            "u": 2,
            "n": 1,
            "empty": 0,
            "universe": 0,
        }[self.operator]
        if len(self.children) != expected_children:
            raise ValueError(
                f"Operator {self.operator!r} requires {expected_children} children, "
                f"got {len(self.children)}"
            )
        if self.operator == "e":
            if not isinstance(self.value, int) or self.value <= 0:
                raise ValueError("Entity leaves require a positive shifted token")
        elif self.operator == "p":
            if not isinstance(self.value, int) or self.value >= 0:
                raise ValueError("Projection nodes require a negative shifted relation token")
        elif self.value is not None:
            raise ValueError(f"Operator {self.operator!r} cannot carry a numeric value")


@dataclass(frozen=True)
class SemanticSlot:
    kind: SlotKind
    value: int
    path: tuple[int, ...]
    ordinal: int
    parent_operator: str | None

    @property
    def path_string(self) -> str:
        location = "root" if not self.path else "/".join(str(index) for index in self.path)
        suffix = "entity" if self.kind == "specific_entity" else "relation"
        return f"{location}:{suffix}"


@dataclass(frozen=True)
class Neutralization:
    query: QueryNode
    branch_operator: str | None
    branch_path: tuple[int, ...]
    identity: str


@dataclass(frozen=True)
class ReplacementCandidate:
    token: int
    fallback: str
    features: Mapping[str, Any]
    denotation: frozenset[int]
    semantic_score: float


def parse_action(action: str | Sequence[str | int]) -> QueryNode:
    """Parse CtrlHGen's prefix action format into an immutable AST."""
    raw_tokens = action.split() if isinstance(action, str) else list(action)
    tokens: list[str | int] = []
    for token in raw_tokens:
        if isinstance(token, int):
            tokens.append(token)
            continue
        try:
            tokens.append(int(token))
        except (TypeError, ValueError):
            tokens.append(str(token))

    def parse_at(index: int) -> tuple[QueryNode, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of action sequence")
        token = tokens[index]
        if token in {"i", "u"}:
            left, next_index = parse_at(index + 1)
            right, next_index = parse_at(next_index)
            return QueryNode(str(token), children=(left, right)), next_index
        if token == "n":
            child, next_index = parse_at(index + 1)
            return QueryNode("n", children=(child,)), next_index
        if not isinstance(token, int) or token == 0:
            raise ValueError(f"Invalid action token at position {index}: {token!r}")
        if token > 0:
            return QueryNode("e", value=token), index + 1
        child, next_index = parse_at(index + 1)
        return QueryNode("p", value=token, children=(child,)), next_index

    query, consumed = parse_at(0)
    if consumed != len(tokens):
        raise ValueError(f"Action sequence has {len(tokens) - consumed} trailing tokens")
    return query


def parse_raw_query(query: Sequence[Any]) -> QueryNode:
    """Parse one sampler JSONL query, whose entity/relation IDs are unshifted."""
    if not isinstance(query, Sequence) or isinstance(query, (str, bytes)):
        raise TypeError("Raw sampler query must be a token sequence")
    shifted = qry_shift_indices(list(query))
    action = qry_str_2_actionstr(list_to_str(shifted))
    return parse_action(action)


def action_tokens(query: QueryNode) -> tuple[str | int, ...]:
    if query.operator == "e":
        return (int(query.value),)
    if query.operator == "p":
        return (int(query.value), *action_tokens(query.children[0]))
    if query.operator in {"i", "u", "n"}:
        result: list[str | int] = [query.operator]
        for child in query.children:
            result.extend(action_tokens(child))
        return tuple(result)
    raise ValueError("Logical identity nodes cannot be serialized as model actions")


def action_string(query: QueryNode) -> str:
    return " ".join(str(token) for token in action_tokens(query))


def shifted_wordlist(query: QueryNode) -> list[str | int]:
    """Serialize a normal AST to the parenthesized shifted query format."""
    if query.operator == "e":
        return ["(", "e", "(", int(query.value), ")", ")"]
    if query.operator == "p":
        return [
            "(", "p", "(", int(query.value), ")",
            *shifted_wordlist(query.children[0]), ")",
        ]
    if query.operator in {"i", "u", "n"}:
        result: list[str | int] = ["(", query.operator]
        for child in query.children:
            result.extend(shifted_wordlist(child))
        result.append(")")
        return result
    raise ValueError("Logical identity nodes cannot be serialized as ordinary queries")


def pattern_signature(query: QueryNode) -> str:
    if query.operator == "e":
        return "e"
    if query.operator == "p":
        return f"p {pattern_signature(query.children[0])}"
    if query.operator in {"i", "u", "n"}:
        children = " ".join(pattern_signature(child) for child in query.children)
        return f"{query.operator} {children}"
    return query.operator.upper()


def slot_counts(query: QueryNode) -> dict[str, int]:
    slots = extract_semantic_slots(query)
    return {
        "specific_entity": sum(slot.kind == "specific_entity" for slot in slots),
        "specific_relation": sum(slot.kind == "specific_relation" for slot in slots),
    }


def extract_semantic_slots(query: QueryNode, kind: SlotKind | None = None) -> list[SemanticSlot]:
    """Return eligible numeric slots in canonical depth-first order."""
    found: list[tuple[SlotKind, int, tuple[int, ...], str | None]] = []

    def visit(node: QueryNode, path: tuple[int, ...], parent: str | None) -> None:
        if node.operator == "p":
            found.append(("specific_relation", int(node.value), path, parent))
        elif node.operator == "e":
            found.append(("specific_entity", int(node.value), path, parent))
        for child_index, child in enumerate(node.children):
            visit(child, (*path, child_index), node.operator)

    visit(query, (), None)
    per_kind: defaultdict[str, int] = defaultdict(int)
    slots: list[SemanticSlot] = []
    for slot_kind, value, path, parent in found:
        ordinal = per_kind[slot_kind]
        per_kind[slot_kind] += 1
        if kind is None or kind == slot_kind:
            slots.append(SemanticSlot(slot_kind, value, path, ordinal, parent))
    return slots


def select_semantic_slot(
    query: QueryNode,
    kind: SlotKind,
    *,
    record_id: str,
    epoch: int,
    seed: int,
) -> SemanticSlot:
    """Select one eligible slot uniformly with a stateless epoch seed."""
    slots = extract_semantic_slots(query, kind)
    if not slots:
        raise ValueError(f"Query has no eligible {kind} slots")
    rng = random.Random(_stable_seed(seed, "sc_idc_slot", record_id, epoch, kind))
    return slots[rng.randrange(len(slots))]


def _node_at(query: QueryNode, path: Sequence[int]) -> QueryNode:
    node = query
    for index in path:
        node = node.children[int(index)]
    return node


def _replace_node(query: QueryNode, path: Sequence[int], replacement: QueryNode) -> QueryNode:
    if not path:
        return replacement
    child_index = int(path[0])
    children = list(query.children)
    children[child_index] = _replace_node(children[child_index], path[1:], replacement)
    return QueryNode(query.operator, value=query.value, children=tuple(children))


def replace_slot(query: QueryNode, slot: SemanticSlot, token: int) -> QueryNode:
    node = _node_at(query, slot.path)
    if slot.kind == "specific_relation":
        if node.operator != "p" or int(token) >= 0:
            raise ValueError("Relation replacement must target p with a negative shifted token")
    else:
        if node.operator != "e" or int(token) <= 0:
            raise ValueError("Entity replacement must target e with a positive shifted token")
    return _replace_node(
        query,
        slot.path,
        QueryNode(node.operator, value=int(token), children=node.children),
    )


def neutralize_slot_branch(query: QueryNode, slot: SemanticSlot) -> Neutralization:
    """Remove the nearest logical branch with its parent's identity element."""
    ancestors: list[tuple[tuple[int, ...], QueryNode]] = []
    node = query
    ancestors.append(((), node))
    for depth, child_index in enumerate(slot.path):
        node = node.children[child_index]
        ancestors.append((tuple(slot.path[: depth + 1]), node))

    for ancestor_path, ancestor in reversed(ancestors[:-1]):
        if ancestor.operator not in SET_OPERATORS:
            continue
        controlled_child_index = int(slot.path[len(ancestor_path)])
        branch_path = (*ancestor_path, controlled_child_index)
        identity = "universe" if ancestor.operator == "i" else "empty"
        return Neutralization(
            query=_replace_node(query, branch_path, QueryNode(identity)),
            branch_operator=ancestor.operator,
            branch_path=branch_path,
            identity=identity,
        )
    return Neutralization(
        query=QueryNode("empty"),
        branch_operator=None,
        branch_path=(),
        identity="empty",
    )


class QueryExecutor:
    """Set executor matching GraphSampler semantics plus two identity nodes."""

    def __init__(self, graph_sampler) -> None:
        self.graph_sampler = graph_sampler
        self.graph = graph_sampler.graph
        self.universe = frozenset(int(node) for node in self.graph.nodes)

    def execute(self, query: QueryNode) -> frozenset[int]:
        operator = query.operator
        if operator == "empty":
            return frozenset()
        if operator == "universe":
            return self.universe
        if operator == "e":
            entity_id = int(query.value) - 1
            return frozenset({entity_id}) if self.graph.has_node(entity_id) else frozenset()
        if operator == "p":
            relation_id = abs(int(query.value)) - 1
            sources = self.execute(query.children[0])
            answers = {
                int(target)
                for _, target, key in self.graph.out_edges(sources, keys=True)
                if int(key) == relation_id
            }
            return frozenset(answers)
        if operator == "i":
            return self.execute(query.children[0]) & self.execute(query.children[1])
        if operator == "u":
            return self.execute(query.children[0]) | self.execute(query.children[1])
        if operator == "n":
            return self.universe - self.execute(query.children[0])
        raise AssertionError(f"Unhandled operator: {operator}")


def _counter_cosine(left: Counter[Any], right: Counter[Any]) -> float:
    common = sorted(set(left) & set(right))
    numerator = math.fsum(float(left[key]) * float(right[key]) for key in common)
    left_norm = math.sqrt(math.fsum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(math.fsum(float(value) ** 2 for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


class MatchedReplacementIndex:
    """Empirical matching index built only from the selected observable graph."""

    def __init__(self, graph_sampler) -> None:
        self.graph_sampler = graph_sampler
        self.graph = graph_sampler.graph
        self.executor = QueryExecutor(graph_sampler)
        self.nodes = tuple(sorted(int(node) for node in self.graph.nodes))
        relation_ids = {int(key) for _, _, key in self.graph.edges(keys=True)}
        self.relation_ids = tuple(sorted(relation_ids))

        self.node_degrees = {
            node: int(self.graph.in_degree(node) + self.graph.out_degree(node))
            for node in self.nodes
        }
        self.node_signatures: dict[int, frozenset[tuple[str, int]]] = {}
        self.node_incidence: dict[int, Counter[tuple[str, int]]] = {}
        self.node_out_relations: dict[int, frozenset[int]] = {}
        for node in self.nodes:
            incidence: Counter[tuple[str, int]] = Counter()
            outgoing = set()
            for _, _, key in self.graph.out_edges(node, keys=True):
                relation_id = int(key)
                incidence[("out", relation_id)] += 1
                outgoing.add(relation_id)
            for _, _, key in self.graph.in_edges(node, keys=True):
                incidence[("in", int(key))] += 1
            self.node_incidence[node] = incidence
            self.node_signatures[node] = frozenset(incidence)
            self.node_out_relations[node] = frozenset(outgoing)

        self.relation_edge_counts = Counter(int(key) for _, _, key in self.graph.edges(keys=True))
        source_nodes: defaultdict[int, set[int]] = defaultdict(set)
        target_nodes: defaultdict[int, set[int]] = defaultdict(set)
        for source, target, key in self.graph.edges(keys=True):
            source_nodes[int(key)].add(int(source))
            target_nodes[int(key)].add(int(target))
        self.relation_head_profiles: dict[int, Counter[Any]] = {}
        self.relation_tail_profiles: dict[int, Counter[Any]] = {}
        for relation_id in self.relation_ids:
            head: Counter[Any] = Counter()
            tail: Counter[Any] = Counter()
            for node in source_nodes[relation_id]:
                head.update(self.node_incidence[node])
            for node in target_nodes[relation_id]:
                tail.update(self.node_incidence[node])
            self.relation_head_profiles[relation_id] = head
            self.relation_tail_profiles[relation_id] = tail

    def relation_direction(self, relation_id: int) -> str:
        name = str(getattr(self.graph_sampler, "id2rel", {}).get(int(relation_id), ""))
        if name.startswith("+"):
            return "forward"
        if name.startswith("-"):
            return "reverse"
        return "unknown"

    def _sample_best(
        self,
        ranked: Sequence[ReplacementCandidate],
        *,
        replacements: int,
        seed: int,
    ) -> list[ReplacementCandidate]:
        best = list(ranked[:8])
        if len(best) <= int(replacements):
            return best
        rng = random.Random(seed)
        selected = rng.sample(best, int(replacements))
        selected.sort(key=lambda candidate: candidate.token)
        return selected

    def _relation_matches(
        self,
        query: QueryNode,
        slot: SemanticSlot,
        observation: frozenset[int],
        base_denotation: frozenset[int],
        *,
        replacements: int,
        seed: int,
    ) -> tuple[list[ReplacementCandidate], int]:
        original_id = abs(slot.value) - 1
        original_direction = self.relation_direction(original_id)
        pool = [
            relation_id
            for relation_id in self.relation_ids
            if relation_id != original_id
            and self.relation_direction(relation_id) == original_direction
        ]
        fallback = "same_direction"
        target_pool_size = min(8, max(0, len(self.relation_ids) - 1))
        if len(pool) < target_pool_size:
            pool = [relation_id for relation_id in self.relation_ids if relation_id != original_id]
            fallback = "all_relations"

        base_size = len(base_denotation)
        base_cardinality_bin = _log_bin(base_size)
        original_frequency_bin = _log_bin(self.relation_edge_counts[original_id])
        ranked: list[tuple[tuple[Any, ...], ReplacementCandidate]] = []
        for relation_id in pool:
            token = -(int(relation_id) + 1)
            candidate_query = replace_slot(query, slot, token)
            denotation = self.executor.execute(candidate_query)
            head_similarity = _counter_cosine(
                self.relation_head_profiles[original_id],
                self.relation_head_profiles[relation_id],
            )
            tail_similarity = _counter_cosine(
                self.relation_tail_profiles[original_id],
                self.relation_tail_profiles[relation_id],
            )
            endpoint_similarity = (head_similarity + tail_similarity) / 2.0
            candidate_cardinality_bin = _log_bin(len(denotation))
            candidate_frequency_bin = _log_bin(self.relation_edge_counts[relation_id])
            card_distance = abs(candidate_cardinality_bin - base_cardinality_bin)
            frequency_distance = abs(candidate_frequency_bin - original_frequency_bin)
            candidate = ReplacementCandidate(
                token=token,
                fallback=fallback,
                features={
                    "direction": self.relation_direction(relation_id),
                    "original_answer_cardinality_log_bin": base_cardinality_bin,
                    "candidate_answer_cardinality_log_bin": candidate_cardinality_bin,
                    "answer_cardinality_bin_distance": card_distance,
                    "original_edge_frequency_log_bin": original_frequency_bin,
                    "candidate_edge_frequency_log_bin": candidate_frequency_bin,
                    "edge_frequency_bin_distance": frequency_distance,
                    "head_signature_cosine": head_similarity,
                    "tail_signature_cosine": tail_similarity,
                    "endpoint_signature_cosine": endpoint_similarity,
                },
                denotation=denotation,
                semantic_score=jaccard_score(denotation, observation),
            )
            ranked.append(
                ((card_distance, frequency_distance, -endpoint_similarity, token), candidate)
            )
        ranked.sort(key=lambda item: item[0])
        selected = self._sample_best(
            [candidate for _, candidate in ranked],
            replacements=replacements,
            seed=seed,
        )
        return selected, len(ranked)

    def _adjacent_relation(self, query: QueryNode, slot: SemanticSlot) -> int | None:
        if not slot.path:
            return None
        parent = _node_at(query, slot.path[:-1])
        if parent.operator != "p":
            return None
        return abs(int(parent.value)) - 1

    def _entity_pool(
        self,
        original_id: int,
        adjacent_relation: int | None,
    ) -> tuple[list[int], str]:
        original_bin = _log_bin(self.node_degrees[original_id])
        reachable = [
            node for node in self.nodes
            if node != original_id
            and (
                adjacent_relation is None
                or adjacent_relation in self.node_out_relations[node]
            )
        ]
        same_bin = [node for node in reachable if _log_bin(self.node_degrees[node]) == original_bin]
        target_pool_size = min(8, max(0, len(self.nodes) - 1))
        if len(same_bin) >= target_pool_size:
            return same_bin, "adjacent_relation_same_degree"
        near_bin = [
            node for node in reachable
            if abs(_log_bin(self.node_degrees[node]) - original_bin) <= 1
        ]
        if len(near_bin) >= target_pool_size:
            return near_bin, "adjacent_relation_near_degree"
        if len(reachable) >= target_pool_size:
            return reachable, "adjacent_relation_any_degree"
        return [node for node in self.nodes if node != original_id], "global"

    def _entity_matches(
        self,
        query: QueryNode,
        slot: SemanticSlot,
        observation: frozenset[int],
        base_denotation: frozenset[int],
        *,
        replacements: int,
        seed: int,
    ) -> tuple[list[ReplacementCandidate], int]:
        original_id = slot.value - 1
        adjacent_relation = self._adjacent_relation(query, slot)
        pool, fallback = self._entity_pool(original_id, adjacent_relation)
        if not pool:
            return [], 0

        hashed_pool = sorted(
            pool,
            key=lambda node: (
                _stable_seed(seed, "entity_prefilter", node),
                node,
            ),
        )[:256]
        original_bin = _log_bin(self.node_degrees[original_id])
        original_signature = self.node_signatures[original_id]
        feature_ranked = sorted(
            hashed_pool,
            key=lambda node: (
                -_jaccard_sets(original_signature, self.node_signatures[node]),
                abs(_log_bin(self.node_degrees[node]) - original_bin),
                abs(self.node_degrees[node] - self.node_degrees[original_id]),
                node,
            ),
        )[:16]

        base_size = len(base_denotation)
        base_cardinality_bin = _log_bin(base_size)
        ranked: list[tuple[tuple[Any, ...], ReplacementCandidate]] = []
        for entity_id in feature_ranked:
            token = int(entity_id) + 1
            candidate_query = replace_slot(query, slot, token)
            denotation = self.executor.execute(candidate_query)
            signature_similarity = _jaccard_sets(
                original_signature, self.node_signatures[entity_id]
            )
            candidate_cardinality_bin = _log_bin(len(denotation))
            candidate_degree_bin = _log_bin(self.node_degrees[entity_id])
            card_distance = abs(candidate_cardinality_bin - base_cardinality_bin)
            degree_bin_distance = abs(candidate_degree_bin - original_bin)
            candidate = ReplacementCandidate(
                token=token,
                fallback=fallback,
                features={
                    "anchor_role": "entity_leaf",
                    "adjacent_relation": adjacent_relation,
                    "adjacent_relation_reachable": (
                        adjacent_relation is None
                        or adjacent_relation in self.node_out_relations[entity_id]
                    ),
                    "original_answer_cardinality_log_bin": base_cardinality_bin,
                    "candidate_answer_cardinality_log_bin": candidate_cardinality_bin,
                    "answer_cardinality_bin_distance": card_distance,
                    "original_degree_log_bin": original_bin,
                    "candidate_degree_log_bin": candidate_degree_bin,
                    "degree_bin_distance": degree_bin_distance,
                    "incident_signature_jaccard": signature_similarity,
                },
                denotation=denotation,
                semantic_score=jaccard_score(denotation, observation),
            )
            ranked.append(
                ((card_distance, degree_bin_distance, -signature_similarity, token), candidate)
            )
        ranked.sort(key=lambda item: item[0])
        selected = self._sample_best(
            [candidate for _, candidate in ranked],
            replacements=replacements,
            seed=seed,
        )
        return selected, len(ranked)

    def matches_with_cost(
        self,
        query: QueryNode,
        slot: SemanticSlot,
        observation: Iterable[int],
        base_denotation: frozenset[int],
        *,
        replacements: int = 3,
        record_id: str,
        seed: int,
    ) -> tuple[list[ReplacementCandidate], int]:
        """Return selected matches and the number of candidate executions."""
        if not 1 <= int(replacements) <= 8:
            raise ValueError("replacements must be between 1 and 8")
        observation_set = frozenset(int(value) for value in observation)
        match_seed = _stable_seed(seed, "sc_idc_match", record_id, slot.path_string)
        if slot.kind == "specific_relation":
            return self._relation_matches(
                query,
                slot,
                observation_set,
                base_denotation,
                replacements=replacements,
                seed=match_seed,
            )
        return self._entity_matches(
            query,
            slot,
            observation_set,
            base_denotation,
            replacements=replacements,
            seed=match_seed,
        )

    def matches(
        self,
        query: QueryNode,
        slot: SemanticSlot,
        observation: Iterable[int],
        *,
        replacements: int = 3,
        record_id: str,
        seed: int,
    ) -> list[ReplacementCandidate]:
        candidates, _ = self.matches_with_cost(
            query,
            slot,
            observation,
            self.executor.execute(query),
            replacements=replacements,
            record_id=record_id,
            seed=seed,
        )
        return candidates


def _path_string(path: Sequence[int]) -> str:
    return "root" if not path else "/".join(str(index) for index in path)


def audit_slot(
    *,
    query: QueryNode,
    observation: Iterable[int],
    slot: SemanticSlot,
    matcher: MatchedReplacementIndex,
    record_id: str,
    seed: int,
    replacements: int = 3,
    epsilon: float = 0.0,
    tau: float = 0.1,
    base_denotation: frozenset[int] | None = None,
) -> dict[str, Any]:
    """Audit one explicit reference slot without changing any training code."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    observation_set = frozenset(int(value) for value in observation)
    base_executed = base_denotation is None
    base = matcher.executor.execute(query) if base_executed else base_denotation
    base_score = jaccard_score(base, observation_set)
    neutralization = neutralize_slot_branch(query, slot)
    neutral = matcher.executor.execute(neutralization.query)
    neutral_score = jaccard_score(neutral, observation_set)
    branch_delta = base_score - neutral_score

    candidates, candidate_execution_count = matcher.matches_with_cost(
        query,
        slot,
        observation_set,
        base,
        replacements=replacements,
        record_id=record_id,
        seed=seed,
    )
    matched_deltas = [base_score - candidate.semantic_score for candidate in candidates]
    matched_delta = (
        sum(matched_deltas) / len(matched_deltas) if matched_deltas else None
    )
    if matched_delta is None:
        classification = "unscorable"
    elif branch_delta <= float(epsilon):
        classification = "laundered"
    elif matched_delta <= float(epsilon):
        classification = "marginal_only"
    else:
        classification = "strict_effective"

    normalized_branch = max(0.0, min(1.0, (branch_delta - epsilon) / tau))
    normalized_matched = (
        0.0
        if matched_delta is None
        else max(0.0, min(1.0, (matched_delta - epsilon) / tau))
    )
    original_pattern = pattern_signature(query)
    original_counts = slot_counts(query)
    candidate_rows = []
    for candidate, delta in zip(candidates, matched_deltas):
        candidate_query = replace_slot(query, slot, candidate.token)
        candidate_rows.append({
            "token": candidate.token,
            "fallback": candidate.fallback,
            "features": dict(candidate.features),
            "semantic_score": candidate.semantic_score,
            "delta": delta,
            "denotation_cardinality": len(candidate.denotation),
            "denotation_sha256": denotation_sha256(candidate.denotation),
            "structure_preserved": (
                pattern_signature(candidate_query) == original_pattern
                and slot_counts(candidate_query) == original_counts
            ),
        })

    return {
        "record_id": str(record_id),
        "condition_kind": slot.kind,
        "condition_value": str(slot.value),
        "slot_ordinal": slot.ordinal,
        "slot_position": "first" if slot.ordinal == 0 else "non_first",
        "slot_path": slot.path_string,
        "slot_parent_operator": slot.parent_operator,
        "nominal_adherence": True,
        "pattern_signature": original_pattern,
        "slot_counts": original_counts,
        "base_semantic_score": base_score,
        "base_denotation_cardinality": len(base),
        "base_denotation_sha256": denotation_sha256(base),
        "reference_exact": base == observation_set,
        "neutralization": {
            "branch_operator": neutralization.branch_operator,
            "branch_path": _path_string(neutralization.branch_path),
            "identity": neutralization.identity,
            "semantic_score": neutral_score,
            "denotation_cardinality": len(neutral),
            "denotation_sha256": denotation_sha256(neutral),
        },
        "branch_marginal_delta": branch_delta,
        "matched_delta": matched_delta,
        "replacement_count": len(candidate_rows),
        "matched_candidate_execution_count": candidate_execution_count,
        "replacements": candidate_rows,
        "classification": classification,
        "diagnostic_score": base_score * normalized_branch * normalized_matched,
        "execution_count": int(base_executed) + 1 + candidate_execution_count,
    }


def legacy_denotation(graph_sampler, query: QueryNode) -> frozenset[int]:
    """Execute a normal AST through the historical GraphSampler for parity tests."""
    from akgr.utils.parsing_util import qry_unshift_indices

    raw = qry_unshift_indices(shifted_wordlist(query))
    return frozenset(int(value) for value in graph_sampler.search_answers_to_query(raw))

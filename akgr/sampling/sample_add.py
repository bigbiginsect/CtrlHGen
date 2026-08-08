"""Deterministic train-only sub-logic augmentation on raw query records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PAPER_AUGMENTATION_PATTERNS = ("up", "3in", "pni", "pin", "inp")
PATTERN_STRING_TO_ABBR = {
    "(p,(u,(p,(e)),(p,(e))))": "up",
    "(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))": "3in",
    "(i,(n,(p,(p,(e)))),(p,(e)))": "pni",
    "(i,(n,(p,(e))),(p,(p,(e))))": "pin",
    "(p,(i,(n,(p,(e))),(p,(e))))": "inp",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()[:20]
    return f"rec-{digest}"


def ensure_record_ids(records: Iterable[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    """Copy records and add stable provenance identifiers when absent."""
    result = []
    for ordinal, source in enumerate(records):
        record = dict(source)
        record.setdefault(
            "record_id",
            _stable_id(split, record.get("pattern_str"), ordinal, record.get("query"), record.get("answers")),
        )
        result.append(record)
    return result


# AST nodes are tuples: ("e", entity), ("p", relation, child),
# ("n", child), and ("i"/"u", child_1, ...).
def parse_query(tokens: Sequence[Any]):
    def parse_at(index: int):
        if index >= len(tokens) or tokens[index] != "(":
            raise ValueError(f"Expected '(' at token {index}")
        operator = tokens[index + 1]
        cursor = index + 2
        if operator == "e":
            if tokens[cursor] != "(" or tokens[cursor + 2] != ")":
                raise ValueError("Malformed entity expression")
            node = ("e", int(tokens[cursor + 1]))
            cursor += 3
        elif operator == "p":
            if tokens[cursor] != "(" or tokens[cursor + 2] != ")":
                raise ValueError("Malformed projection relation")
            relation = int(tokens[cursor + 1])
            child, cursor = parse_at(cursor + 3)
            node = ("p", relation, child)
        elif operator == "n":
            child, cursor = parse_at(cursor)
            node = ("n", child)
        elif operator in {"i", "u"}:
            children = []
            while cursor < len(tokens) and tokens[cursor] == "(":
                child, cursor = parse_at(cursor)
                children.append(child)
            if len(children) < 2:
                raise ValueError(f"{operator} expression needs at least two children")
            node = (operator, *children)
        else:
            raise ValueError(f"Unsupported query operator: {operator!r}")
        if cursor >= len(tokens) or tokens[cursor] != ")":
            raise ValueError(f"Expected ')' at token {cursor}")
        return node, cursor + 1

    node, end = parse_at(0)
    if end != len(tokens):
        raise ValueError(f"Trailing query tokens beginning at {end}")
    return node


def render_query(node) -> list[Any]:
    operator = node[0]
    if operator == "e":
        return ["(", "e", "(", int(node[1]), ")", ")"]
    if operator == "p":
        return ["(", "p", "(", int(node[1]), ")", *render_query(node[2]), ")"]
    if operator == "n":
        return ["(", "n", *render_query(node[1]), ")"]
    if operator in {"i", "u"}:
        rendered: list[Any] = ["(", operator]
        for child in node[1:]:
            rendered.extend(render_query(child))
        rendered.append(")")
        return rendered
    raise ValueError(f"Unsupported AST operator: {operator!r}")


def render_pattern(node) -> str:
    operator = node[0]
    if operator == "e":
        return "(e)"
    if operator == "p":
        return f"(p,{render_pattern(node[2])})"
    if operator == "n":
        return f"(n,{render_pattern(node[1])})"
    if operator in {"i", "u"}:
        return f"({operator},{','.join(render_pattern(child) for child in node[1:])})"
    raise ValueError(f"Unsupported AST operator: {operator!r}")


def _expect(node, operator: str, arity: int | None = None):
    if node[0] != operator or (arity is not None and len(node) - 1 != arity):
        raise ValueError(f"Expected {operator}/{arity}, got {render_pattern(node)}")
    return node


def decompose_query(node, pattern_abbr: str, graph) -> list[tuple[str, Any]]:
    """Return named concrete subqueries matching the authors' five transforms."""
    if pattern_abbr == "up":
        _expect(node, "p")
        union = _expect(node[2], "u", 2)
        return [("up_to_2p", ("p", node[1], child)) for child in union[1:]]

    if pattern_abbr == "3in":
        outer = _expect(node, "i", 2)
        inner = _expect(outer[1], "i", 2)
        negative = _expect(inner[1], "n")[1]
        positive_1, positive_2 = inner[2], outer[2]
        return [
            ("3in_to_1p", positive_1),
            ("3in_to_1p", positive_2),
            ("3in_to_2in", ("i", ("n", negative), positive_1)),
            ("3in_to_2in", ("i", ("n", negative), positive_2)),
            ("3in_to_2i", ("i", positive_1, positive_2)),
        ]

    if pattern_abbr == "pni":
        intersection = _expect(node, "i", 2)
        _expect(intersection[1], "n")
        return [("pni_to_1p", intersection[2])]

    if pattern_abbr == "pin":
        intersection = _expect(node, "i", 2)
        _expect(intersection[1], "n")
        return [("pin_to_2p", intersection[2])]

    if pattern_abbr == "inp":
        _expect(node, "p")
        intersection = _expect(node[2], "i", 2)
        _expect(intersection[1], "n")
        positive_chain = ("p", node[1], intersection[2])
        # The authors' script also turns each concrete intermediate entity of
        # the positive branch into a one-hop query under the outer relation.
        intermediates = sorted(graph.search_answers_to_query(render_query(intersection[2])))
        one_hops = [("p", node[1], ("e", int(entity))) for entity in intermediates]
        return [("inp_to_1p", query) for query in one_hops] + [("inp_to_2p", positive_chain)]

    raise ValueError(f"Pattern {pattern_abbr!r} is not an augmentation source")


def augment_records(
    records: Iterable[Mapping[str, Any]],
    graph,
    *,
    max_answers: int = 32,
    source_patterns: Iterable[str] = PAPER_AUGMENTATION_PATTERNS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create deterministic, complete raw records using only the supplied graph."""
    allowed = set(source_patterns)
    if not allowed.issubset(PAPER_AUGMENTATION_PATTERNS):
        raise ValueError(f"Unsupported augmentation source patterns: {sorted(allowed)}")
    parents = ensure_record_ids(records, "train")
    augmented: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source_counts = {pattern: 0 for pattern in PAPER_AUGMENTATION_PATTERNS}
    generated_source_counts = {pattern: 0 for pattern in PAPER_AUGMENTATION_PATTERNS}
    output_pattern_counts: dict[str, int] = {}

    for parent in parents:
        pattern_abbr = PATTERN_STRING_TO_ABBR.get(parent.get("pattern_str"))
        if pattern_abbr not in allowed:
            continue
        source_counts[pattern_abbr] += 1
        try:
            node = parse_query(parent["query"])
            candidates = decompose_query(node, pattern_abbr, graph)
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"parent_record_id": parent["record_id"], "reason": f"parse/decompose: {exc}"})
            continue

        seen: set[str] = set()
        for subquery_ordinal, (augmentation_type, subquery) in enumerate(candidates):
            query = render_query(subquery)
            query_key = _canonical_json(query)
            if query_key in seen:
                skipped.append({"parent_record_id": parent["record_id"], "reason": "duplicate subquery"})
                continue
            seen.add(query_key)
            answers = sorted(int(value) for value in graph.search_answers_to_query(query))
            if not answers or len(answers) > int(max_answers):
                skipped.append(
                    {
                        "parent_record_id": parent["record_id"],
                        "reason": "empty answers" if not answers else "answers exceed max_answers",
                        "answer_count": len(answers),
                    }
                )
                continue
            pattern_str = render_pattern(subquery)
            record = {
                "answers": answers,
                "query": query,
                "pattern_str": pattern_str,
                "record_id": _stable_id(parent["record_id"], subquery_ordinal, query),
                "parent_record_id": parent["record_id"],
                "parent_pattern": parent["pattern_str"],
                "augmentation_type": augmentation_type,
                "subquery_ordinal": subquery_ordinal,
            }
            augmented.append(record)
            generated_source_counts[pattern_abbr] += 1
            output_pattern_counts[pattern_str] = output_pattern_counts.get(pattern_str, 0) + 1

    report = {
        "source_counts": source_counts,
        "generated_source_counts": generated_source_counts,
        "output_pattern_counts": dict(sorted(output_pattern_counts.items())),
        "generated_count": len(augmented),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    return augmented, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Base train JSONL")
    parser.add_argument("--output", type=Path, help="Augmented-only JSONL")
    parser.add_argument("-d", "--dataname", default="WN18RR")
    parser.add_argument("--data-root", "--data_root", dest="data_root", default="./sampled_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-a", "--max-answer-size", type=int, default=32)
    parser.add_argument("--scale", default="debug")
    # Historical training-shaped arguments are accepted so the authors' shell
    # wrapper still reaches the raw-record transform. They do not affect data.
    parser.add_argument("--modelname")
    parser.add_argument("--config-dataloader")
    parser.add_argument("--config-train")
    parser.add_argument("--config-model")
    parser.add_argument("--config-batchsize")
    parser.add_argument("--checkpoint_root")
    parser.add_argument("-r", "--resume_epoch", type=int)
    parser.add_argument("--result_root")
    parser.add_argument("--save_frequency", type=int)
    parser.add_argument("--mode")
    args = parser.parse_args()
    from akgr.kgdata import load_kg

    data_root = Path(args.data_root).expanduser().resolve()
    base_name = f"{args.dataname}-{args.scale}-{args.max_answer_size}-train"
    input_path = args.input or data_root / args.dataname / f"{base_name}-a2q.jsonl"
    output_path = args.output or data_root / args.dataname / f"{base_name}-augmented-only-a2q.jsonl"
    kg = load_kg(args.dataname, data_root=args.data_root, seed=args.seed)
    augmented, report = augment_records(
        _read_jsonl(input_path), kg.graph_samplers["train"], max_answers=args.max_answer_size
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, augmented)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

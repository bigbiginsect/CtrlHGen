import csv
import json

from akgr.utils.parsing_util import (
    list_to_str,
    qry_actionstr_2_wordlist,
    qry_shift_indices,
    qry_str_2_actionstr,
    qry_unshift_indices,
)


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
        # Raw KG queries encode relation IDs as non-positive integers.
        relation = -counters["relation"]
        counters["relation"] += 1
        return ["(", "p", "(", relation, ")", *_concrete_query(children[0], counters), ")"]
    if operator == "n":
        return ["(", "n", *_concrete_query(children[0], counters), ")"]
    if operator in {"i", "u"}:
        result = ["(", operator]
        for child in children:
            result.extend(_concrete_query(child, counters))
        return [*result, ")"]
    raise AssertionError(f"unsupported test operator: {operator}")


def test_all_thirteen_reproduction_patterns_round_trip_through_actions():
    with open("akgr/metadata/pattern_filtered.csv", newline="", encoding="utf-8") as handle:
        patterns = list(csv.DictReader(handle))
    assert len(patterns) == 13

    for row in patterns:
        concrete = _concrete_query(
            _parse_pattern(row["pattern_str"]), {"entity": 0, "relation": 0}
        )
        shifted = qry_shift_indices(concrete)
        action = qry_str_2_actionstr(list_to_str(shifted))
        reconstructed_shifted = qry_actionstr_2_wordlist(action)
        assert reconstructed_shifted == shifted, row["pattern_abbr"]
        assert qry_unshift_indices(reconstructed_shifted) == concrete, row["pattern_abbr"]

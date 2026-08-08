import networkx as nx

from akgr.kgdata.kgclass import GraphSampler
from akgr.sampling.sample_add import augment_records, parse_query, render_pattern, render_query


def e(value):
    return ("e", value)


def p(relation, child):
    return ("p", -relation, child)


def n(child):
    return ("n", child)


def i(*children):
    return ("i", *children)


def u(*children):
    return ("u", *children)


def _sampler():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, key=1)
    graph.add_edge(2, 1, key=2)
    graph.add_edge(6, 0, key=3)
    graph.add_edge(1, 4, key=0)
    return GraphSampler(graph, {key: f"+r{key}" for key in range(4)})


def _record(node):
    return {"answers": [4], "query": render_query(node), "pattern_str": render_pattern(node)}


def test_five_paper_patterns_produce_complete_train_records_deterministically():
    positive_1, positive_2, negative = p(1, e(0)), p(2, e(2)), p(3, e(6))
    records = [
        _record(p(0, u(positive_1, positive_2))),
        _record(i(i(n(negative), positive_1), positive_2)),
        _record(i(n(p(1, positive_1)), positive_2)),
        _record(i(n(negative), p(0, positive_1))),
        _record(p(0, i(n(negative), positive_2))),
    ]
    first, report = augment_records(records, _sampler(), max_answers=32)
    second, _ = augment_records(records, _sampler(), max_answers=32)

    assert first == second
    assert report["source_counts"] == {"up": 1, "3in": 1, "pni": 1, "pin": 1, "inp": 1}
    assert {record["augmentation_type"].split("_to_")[0] for record in first} == {"up", "3in", "pni", "pin", "inp"}
    assert all(record["answers"] for record in first)
    assert all("parent_record_id" in record and "subquery_ordinal" in record for record in first)
    assert all(record["pattern_str"] == render_pattern(parse_query(record["query"])) for record in first)


def test_non_target_pattern_is_not_augmented():
    records = [_record(p(1, e(0)))]
    augmented, report = augment_records(records, _sampler())
    assert augmented == []
    assert report["generated_count"] == 0

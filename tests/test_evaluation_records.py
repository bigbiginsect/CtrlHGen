import json

import pytest

pytest.importorskip("smatch")

from akgr.evaluation import (
    build_evaluation_record,
    condition_accuracy,
    scoring_input_act_batch,
    scoring_input_act_batch_condition,
    write_evaluation_jsonl,
)
from akgr.reproduction.contracts import ConditionSpec


@pytest.mark.parametrize(
    ("spec", "good", "bad"),
    [
        (ConditionSpec("pattern", "i p e p e"), "i -1 1 -2 2", "-1 1"),
        (ConditionSpec("entity_number", "2e"), "i -1 1 -2 2", "-1 1"),
        (ConditionSpec("relation_number", "2p"), "i -1 1 -2 2", "-1 1"),
        (ConditionSpec("specific_entity", "2"), "i -1 1 -2 2", "-1 1"),
        (ConditionSpec("specific_relation", "-2"), "i -1 1 -2 2", "-1 1"),
    ],
)
def test_condition_accuracy_uses_explicit_request(spec, good, bad):
    assert condition_accuracy(good, spec) == 1.0
    assert condition_accuracy(bad, spec) == 0.0
    assert condition_accuracy(good, None) is None


def test_record_separates_parse_failure_from_zero_scores(tmp_path):
    record = build_evaluation_record(
        record_id="test:0", observation="1", reference="-1 1", prediction="garbage",
        scores={"jaccard": 0, "dice": 0, "overlap": 0, "smatch": 0}, condition=None,
    )
    assert record["parse_ok"] is False
    assert record["condition_accuracy"] is None
    path = write_evaluation_jsonl(tmp_path / "records.jsonl", [record])
    assert json.loads(path.read_text().strip())["record_id"] == "test:0"


@pytest.mark.parametrize(
    "spec",
    [
        ConditionSpec("pattern", "i p e p e"),
        ConditionSpec("entity_number", "2e"),
        ConditionSpec("relation_number", "2p"),
        ConditionSpec("specific_entity", "2"),
        ConditionSpec("specific_relation", "-2"),
    ],
)
def test_conditional_record_contract(spec):
    record = build_evaluation_record(
        record_id="test:1", observation="1 2", reference="i -1 1 -2 2",
        prediction="i -1 1 -2 2",
        scores={"jaccard": 1, "dice": 1, "overlap": 1, "smatch": 1},
        condition=spec,
    )
    assert record == {
        "record_id": "test:1",
        "observation": "1 2",
        "condition": {"kind": spec.kind, "value": spec.value},
        "reference": "i -1 1 -2 2",
        "prediction": "i -1 1 -2 2",
        "jaccard": 1.0,
        "dice": 1.0,
        "overlap": 1.0,
        "condition_accuracy": 1.0,
        "smatch": 1.0,
        "parse_ok": True,
        "parse_error": None,
    }


def test_unconditional_record_contract():
    record = build_evaluation_record(
        record_id="test:2", observation="1", reference="-1 1", prediction="-1 1",
        scores={}, condition=None,
    )
    assert record["condition"] is None
    assert record["condition_accuracy"] is None
    assert record["parse_ok"] is True


class _ExactAnswerSampler:
    def search_answers_to_query(self, query):
        return [0]


def _score_one_batch(spec):
    common = dict(
        pred_word_batch=["-1 1"],
        label_word_batch=["-1 1"],
        ans_word_batch=["1"],
        graph_samplers={"test": _ExactAnswerSampler()},
        searching_split="test",
        return_failures=True,
    )
    methods = ["smatch", "jaccard", "dice", "overlap"]
    if spec is None:
        return scoring_input_act_batch(scoring_method=methods, **common)[0][0]
    condition_method = "specific" if spec.kind.startswith("specific_") else "validity"
    return scoring_input_act_batch_condition(
        condition_batch=[spec.value], scoring_method=methods + [condition_method], **common
    )[0][0]


@pytest.mark.parametrize(
    "spec",
    [
        None,
        ConditionSpec("pattern", "p e"),
        ConditionSpec("entity_number", "1e"),
        ConditionSpec("relation_number", "1p"),
        ConditionSpec("specific_entity", "1"),
        ConditionSpec("specific_relation", "-1"),
    ],
)
def test_unconditional_and_five_conditional_single_batch_evaluation(spec):
    scores = _score_one_batch(spec)
    record = build_evaluation_record(
        record_id="test:batch", observation="1", reference="-1 1",
        prediction="-1 1", scores=scores, condition=spec,
    )
    assert record["parse_ok"] is True
    assert record["smatch"] == pytest.approx(1.0)
    assert record["jaccard"] == pytest.approx(1.0)
    assert record["dice"] == pytest.approx(1.0)
    assert record["overlap"] == pytest.approx(1.0, abs=1e-4)
    assert record["condition_accuracy"] == (None if spec is None else 1.0)

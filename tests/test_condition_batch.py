import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from akgr.tokenizer import create_reproduction_tokenizer, create_tokenizer, prepare_batch


SPECIAL = {
    "PAD": 0, "START": 2, "END": 1, "UNK": 3, "SEP": 4,
    "(": 10, ")": 11, "e": 13, "p": 14, "i": 15, "u": 16, "n": 17,
    "1p": 18, "2p": 19, "3p": 20, "4p": 21,
    "1e": 22, "2e": 23, "3e": 24, "4e": 25, "5e": 26, "with": 27,
}


@pytest.fixture()
def tokenizer():
    return create_reproduction_tokenizer(nentity=8, nrelation=8)


def test_legacy_tokenizer_remains_available_with_historical_ids():
    legacy = create_tokenizer(SPECIAL, offset=100, nentity=8, nrelation=8, is_gpt=True)[0]
    assert legacy.convert_tokens_to_ids("1") == 100


@pytest.fixture()
def sample():
    return {"source": ["1 2"], "target": ["i -1 1 -2 2"], "pattern_id": [0]}


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("pattern", "i p e p e"),
        ("relation_number", "2p"),
        ("entity_number", "2e"),
        ("specific_relation", "-1"),
        ("specific_entity", "1"),
    ],
)
@pytest.mark.parametrize("is_gen", [False, True])
def test_condition_dispatcher_uses_real_separator_in_train_and_generation(
    tokenizer, sample, kind, value, is_gen
):
    batch = prepare_batch("cpu", sample, tokenizer, True, 32, 16, is_gen, kind)
    assert batch.conditions[0].kind == kind
    assert batch.conditions[0].value == value
    ids = batch.input_ids[0].tolist()
    assert tokenizer.sep_token_id in ids
    assert tokenizer.unk_token_id not in ids
    assert batch.input_ids.shape == batch.attention_mask.shape
    assert batch.labels.ndim == batch.input_ids.ndim == 2
    assert (batch.labels == -100).any()
    if is_gen:
        assert batch.source_attention_mask.shape == batch.input_ids.shape
    else:
        assert batch.labels.shape == batch.input_ids.shape


@pytest.mark.parametrize("is_gen", [False, True])
def test_unconditional_has_fixed_prepared_batch_shape_in_train_and_generation(
    tokenizer, sample, is_gen
):
    batch = prepare_batch("cpu", sample, tokenizer, True, 32, 16, is_gen, "unconditional")
    assert batch.conditions is None
    assert batch.input_ids.shape == batch.attention_mask.shape
    assert (batch.labels == -100).any()
    if is_gen:
        assert batch.source_attention_mask.shape == batch.input_ids.shape
    else:
        assert batch.labels.shape == batch.input_ids.shape

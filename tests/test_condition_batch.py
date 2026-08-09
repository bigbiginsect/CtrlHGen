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
def test_condition_dispatcher_preserves_target_boundary_in_train_and_generation(
    tokenizer, sample, kind, value, is_gen
):
    batch = prepare_batch(
        "cpu", sample, tokenizer, True, 32, 16, is_gen, kind,
        condition_delimiter="COND",
    )
    assert batch.conditions[0].kind == kind
    assert batch.conditions[0].value == value
    ids = batch.input_ids[0].tolist()
    active_ids = [token_id for token_id in ids if token_id != tokenizer.pad_token_id]
    active_tokens = tokenizer.convert_ids_to_tokens(active_ids)
    assert active_tokens.count("COND") == 1
    assert active_tokens.count("SEP") == 1
    assert active_tokens[:3] == ["1", "2", "COND"]
    assert active_tokens[3:3 + len(value.split())] == value.split()
    assert tokenizer.unk_token_id not in ids
    assert batch.input_ids.shape == batch.attention_mask.shape
    assert batch.labels.ndim == batch.input_ids.ndim == 2
    assert (batch.labels == -100).any()
    if is_gen:
        assert active_tokens[-1] == "SEP"
        assert batch.source_attention_mask.shape == batch.input_ids.shape
    else:
        assert active_tokens[-6:] == ["i", "-1", "1", "-2", "2", "END"]
        trained_tokens = tokenizer.convert_ids_to_tokens(
            [token_id for token_id in batch.labels[0].tolist() if token_id != -100]
        )
        assert trained_tokens == ["i", "-1", "1", "-2", "2", "END"]
        assert batch.labels.shape == batch.input_ids.shape


@pytest.mark.parametrize("is_gen", [False, True])
def test_unconditional_has_fixed_prepared_batch_shape_in_train_and_generation(
    tokenizer, sample, is_gen
):
    batch = prepare_batch("cpu", sample, tokenizer, True, 32, 16, is_gen, "unconditional")
    assert batch.conditions is None
    active_tokens = tokenizer.convert_ids_to_tokens(
        [token_id for token_id in batch.input_ids[0].tolist() if token_id != tokenizer.pad_token_id]
    )
    assert "COND" not in active_tokens
    assert active_tokens.count("SEP") == 1
    assert batch.input_ids.shape == batch.attention_mask.shape
    assert (batch.labels == -100).any()
    if is_gen:
        assert batch.source_attention_mask.shape == batch.input_ids.shape
    else:
        assert batch.labels.shape == batch.input_ids.shape


def test_training_forces_right_padding_even_if_loaded_tokenizer_was_left_padded(tokenizer):
    tokenizer.padding_side = "left"
    varied = {
        "source": ["1", "1 2 3 4"],
        "target": ["-1 1", "i -1 1 -2 2"],
        "pattern_id": [0, 1],
    }
    batch = prepare_batch(
        "cpu", varied, tokenizer, True, 32, 16, False, "pattern",
        condition_delimiter="COND",
    )

    assert tokenizer.padding_side == "left"
    assert batch.input_ids[0, 0] != tokenizer.pad_token_id
    assert batch.input_ids[0, -1] == tokenizer.pad_token_id
    for row, target in enumerate(varied["target"]):
        active_labels = [value for value in batch.labels[row].tolist() if value != -100]
        assert tokenizer.convert_ids_to_tokens(active_labels) == target.split() + ["END"]

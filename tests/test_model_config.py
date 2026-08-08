import pytest

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from akgr.abduction_model.transformer import create_gpt2_config, create_reproduction_transformer
from akgr.tokenizer import create_reproduction_tokenizer


class StubTokenizer:
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 1

    def __len__(self):
        return 128

    def get_vocab(self):
        return {str(index): index for index in range(128)}


def test_explicit_gpt2_config_is_offline_and_strict():
    config = create_gpt2_config(StubTokenizer(), {
        "n_layer": 12, "n_embd": 768, "n_head": 12,
        "n_positions": 1024, "n_ctx": 1024,
    })
    assert (config.n_layer, config.n_embd, config.n_head) == (12, 768, 12)
    assert config.vocab_size == 128
    assert config._name_or_path == ""


def test_explicit_gpt2_config_rejects_incomplete_or_invalid_shape():
    with pytest.raises(ValueError, match="Missing"):
        create_gpt2_config(StubTokenizer(), {"n_layer": 12})
    with pytest.raises(ValueError, match="divisible"):
        create_gpt2_config(StubTokenizer(), {
            "n_layer": 2, "n_embd": 63, "n_head": 8, "n_positions": 64, "n_ctx": 64,
        })


def test_real_reproduction_tokenizer_is_contiguous_and_runs_tiny_forward():
    tokenizer = create_reproduction_tokenizer(nentity=8, nrelation=4)
    vocab = tokenizer.get_vocab()
    assert sorted(vocab.values()) == list(range(len(tokenizer)))
    assert len(tokenizer) == len(vocab)
    model = create_reproduction_transformer(tokenizer, {
        "n_layer": 1, "n_embd": 32, "n_head": 4,
        "n_positions": 64, "n_ctx": 64,
    })
    encoded = tokenizer("1 2", "-1 1", return_tensors="pt")
    output = model(**encoded, labels=encoded.input_ids)
    assert output.logits.shape == (1, encoded.input_ids.shape[1], len(tokenizer))
    assert torch.isfinite(output.loss)


@pytest.mark.synthetic
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_reproduction_gpt2_config_initializes_and_forwards_offline_on_gpu():
    tokenizer = create_reproduction_tokenizer(nentity=8, nrelation=4)
    model = create_reproduction_transformer(tokenizer, {
        "n_layer": 12, "n_embd": 768, "n_head": 12,
        "n_positions": 1024, "n_ctx": 1024,
    }).to("cuda")
    encoded = tokenizer("1 2", "-1 1", return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**encoded, labels=encoded.input_ids)
    assert output.logits.shape == (1, encoded.input_ids.shape[1], len(tokenizer))
    assert torch.isfinite(output.loss)

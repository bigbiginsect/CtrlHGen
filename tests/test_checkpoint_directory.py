import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from akgr.tokenizer import create_reproduction_tokenizer
from akgr.utils.load_util import load_reproduction_checkpoint, save_reproduction_checkpoint


SPECIAL = {
    "PAD": 0, "START": 2, "END": 1, "UNK": 3, "SEP": 4,
    "(": 10, ")": 11, "e": 13, "p": 14, "i": 15, "u": 16, "n": 17,
    "1p": 18, "2p": 19, "3p": 20, "4p": 21,
    "1e": 22, "2e": 23, "3e": 24, "4e": 25, "5e": 26, "with": 27,
}


def _objects():
    tokenizer = create_reproduction_tokenizer(8, 8)
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_embd=32, n_head=4,
        n_positions=64, n_ctx=64, pad_token_id=0, bos_token_id=2, eos_token_id=1,
    )
    model = transformers.GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, tokenizer, optimizer, scheduler


def test_checkpoint_parent_resume_and_test_semantics(tmp_path):
    model, tokenizer, optimizer, scheduler = _objects()
    target = tmp_path / "checkpoint"
    save_reproduction_checkpoint(
        target, model=model, tokenizer=tokenizer,
        stage="unconditional", stage_epoch=1, global_step=3,
        condition="unconditional", experiment_config={"test": True},
        experiment_config_hash="abc", seed=42,
        optimizer=optimizer, scheduler=scheduler,
    )
    loaded = load_reproduction_checkpoint(target, mode="test")
    assert loaded.metadata["global_step"] == 3
    assert loaded.model.config.n_layer == 1
    assert (target / "model" / "model.safetensors").is_file()
    assert not (target / "model_state.pt").exists()

    resumed_optimizer = torch.optim.AdamW(loaded.model.parameters(), lr=9e-4)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    resumed = load_reproduction_checkpoint(
        target, mode="resume", model=loaded.model,
        optimizer=resumed_optimizer, scheduler=resumed_scheduler,
        expected_stage="unconditional", expected_condition="unconditional",
        expected_config_hash="abc",
    )
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert resumed.metadata["stage_epoch"] == 1


def test_resume_rejects_wrong_stage_or_missing_optimizer(tmp_path):
    model, tokenizer, optimizer, scheduler = _objects()
    target = tmp_path / "checkpoint"
    save_reproduction_checkpoint(
        target, model=model, tokenizer=tokenizer, stage="conditional", stage_epoch=1,
        global_step=1, condition="pattern", experiment_config={},
        experiment_config_hash="abc", seed=42, optimizer=optimizer, scheduler=scheduler,
    )
    with pytest.raises(ValueError, match="stage"):
        load_reproduction_checkpoint(target, mode="test", expected_stage="unconditional")
    with pytest.raises(ValueError, match="optimizer"):
        load_reproduction_checkpoint(target, mode="resume")

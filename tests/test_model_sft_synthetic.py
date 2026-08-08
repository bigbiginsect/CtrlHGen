from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("smatch")

from akgr.abduction_model.reproduction import create_sft_optimizer_schedule, sft_train_epoch
from akgr.tokenizer import create_reproduction_tokenizer
from akgr.utils.load_util import load_reproduction_checkpoint, save_reproduction_checkpoint


def _model(tokenizer):
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_embd=24, n_head=4,
        n_positions=32, n_ctx=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return transformers.GPT2LMHeadModel(config)


def _batches():
    return [
        {"input_ids": torch.tensor([[2, 5, 8, 1]]), "labels": torch.tensor([[-100, 5, 8, 1]])},
        {"input_ids": torch.tensor([[2, 6, 9, 1]]), "labels": torch.tensor([[-100, 6, 9, 1]])},
    ]


def _prepare(sample):
    return SimpleNamespace(
        input_ids=sample["input_ids"],
        attention_mask=torch.ones_like(sample["input_ids"]),
        labels=sample["labels"],
    )


def _schedule(model):
    return create_sft_optimizer_schedule(
        model, learning_rate=1e-3, num_batches=2,
        gradient_accumulation_steps=1, epochs=2, warmup_epochs=0,
    )


@pytest.mark.synthetic
def test_sft_train_epoch_updates_parameters():
    torch.manual_seed(7)
    tokenizer = create_reproduction_tokenizer(4, 2)
    model = _model(tokenizer)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    schedule = _schedule(model)
    loss, steps = sft_train_epoch(
        model=model, dataloader=_batches(), prepare=_prepare,
        optimizer=schedule.optimizer, scheduler=schedule.scheduler,
        gradient_accumulation_steps=1,
    )
    assert loss > 0
    assert steps == 2
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.synthetic
def test_two_epochs_equal_checkpoint_resume(tmp_path):
    torch.manual_seed(11)
    tokenizer = create_reproduction_tokenizer(4, 2)
    seed_model = _model(tokenizer)
    initial_state = deepcopy(seed_model.state_dict())

    continuous = _model(tokenizer)
    continuous.load_state_dict(initial_state)
    continuous_schedule = _schedule(continuous)
    for _ in range(2):
        sft_train_epoch(
            model=continuous, dataloader=_batches(), prepare=_prepare,
            optimizer=continuous_schedule.optimizer, scheduler=continuous_schedule.scheduler,
            gradient_accumulation_steps=1,
        )

    interrupted = _model(tokenizer)
    interrupted.load_state_dict(initial_state)
    interrupted_schedule = _schedule(interrupted)
    sft_train_epoch(
        model=interrupted, dataloader=_batches(), prepare=_prepare,
        optimizer=interrupted_schedule.optimizer, scheduler=interrupted_schedule.scheduler,
        gradient_accumulation_steps=1,
    )
    checkpoint = tmp_path / "epoch-1"
    save_reproduction_checkpoint(
        checkpoint, model=interrupted, tokenizer=tokenizer,
        stage="unconditional", stage_epoch=1, global_step=2,
        condition="unconditional", experiment_config={},
        experiment_config_hash="synthetic", seed=11,
        optimizer=interrupted_schedule.optimizer, scheduler=interrupted_schedule.scheduler,
    )

    resumed = _model(tokenizer)
    resumed_schedule = _schedule(resumed)
    load_reproduction_checkpoint(
        checkpoint, mode="resume", model=resumed,
        optimizer=resumed_schedule.optimizer, scheduler=resumed_schedule.scheduler,
        expected_stage="unconditional", expected_condition="unconditional",
        expected_config_hash="synthetic", restore_rng=True,
    )
    sft_train_epoch(
        model=resumed, dataloader=_batches(), prepare=_prepare,
        optimizer=resumed_schedule.optimizer, scheduler=resumed_schedule.scheduler,
        gradient_accumulation_steps=1,
    )
    for name, expected in continuous.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)

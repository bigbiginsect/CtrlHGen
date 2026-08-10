import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("smatch")

from akgr.abduction_model.reproduction import create_sft_optimizer_schedule


AUTHOR_OPTIMIZER = {
    "name": "adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
}
AUTHOR_SCHEDULER = {
    "name": "linear_warmup_constant",
    "warmup_unit": "optimizer_step",
    "warmup_value": 5,
    "start_factor": 0.1,
}


def test_adamw_warmup_is_counted_in_optimizer_steps():
    model = torch.nn.Linear(2, 2)
    result = create_sft_optimizer_schedule(
        model, learning_rate=1e-5, num_batches=9,
        gradient_accumulation_steps=4, epochs=3, warmup_epochs=1,
    )
    assert isinstance(result.optimizer, torch.optim.AdamW)
    assert result.optimizer_steps_per_epoch == 3
    assert result.warmup_steps == 3
    assert result.total_steps == 9
    assert result.metadata["style"] == "legacy"
    assert result.metadata["scheduler"] == "linear_warmup_decay"


def test_legacy_adamw_schedule_still_warms_then_decays_to_zero():
    model = torch.nn.Linear(2, 2)
    result = create_sft_optimizer_schedule(
        model, learning_rate=1e-5, num_batches=2,
        gradient_accumulation_steps=1, epochs=2, warmup_epochs=1,
    )
    learning_rates = [result.optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        result.optimizer.step()
        result.scheduler.step()
        learning_rates.append(result.optimizer.param_groups[0]["lr"])
    assert learning_rates == pytest.approx([0.0, 0.5e-5, 1e-5, 0.5e-5, 0.0])


def test_author_aligned_adam_warms_for_five_steps_then_stays_constant():
    model = torch.nn.Linear(2, 2)
    result = create_sft_optimizer_schedule(
        model, learning_rate=5e-5, num_batches=9,
        gradient_accumulation_steps=1, epochs=3,
        optimizer_config=AUTHOR_OPTIMIZER,
        scheduler_config=AUTHOR_SCHEDULER,
    )
    assert type(result.optimizer) is torch.optim.Adam
    assert result.optimizer.defaults["betas"] == (0.9, 0.999)
    assert result.optimizer.defaults["eps"] == 1e-8
    assert result.optimizer.defaults["weight_decay"] == 0.0
    assert result.optimizer_steps_per_epoch == 9
    assert result.warmup_steps == 5
    assert result.total_steps == 27
    assert result.metadata == {
        "style": "explicit",
        "optimizer": "adam",
        "scheduler": "linear_warmup_constant",
        "warmup_unit": "optimizer_step",
        "warmup_value": 5,
        "warmup_steps": 5,
        "start_factor": 0.1,
        "optimizer_steps_per_epoch": 9,
        "total_steps": 27,
    }

    learning_rates = [result.optimizer.param_groups[0]["lr"]]
    for _ in range(8):
        result.optimizer.step()
        result.scheduler.step()
        learning_rates.append(result.optimizer.param_groups[0]["lr"])
    expected_factors = [0.1, 0.28, 0.46, 0.64, 0.82, 1.0, 1.0, 1.0, 1.0]
    assert learning_rates == pytest.approx([5e-5 * factor for factor in expected_factors])


def test_explicit_schedule_rejects_warmup_longer_than_training():
    model = torch.nn.Linear(2, 2)
    scheduler = dict(AUTHOR_SCHEDULER, warmup_value=7)
    with pytest.raises(ValueError, match="cannot exceed"):
        create_sft_optimizer_schedule(
            model, learning_rate=5e-5, num_batches=3,
            gradient_accumulation_steps=1, epochs=2,
            optimizer_config=AUTHOR_OPTIMIZER,
            scheduler_config=scheduler,
        )

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("smatch")

from akgr.abduction_model.reproduction import create_sft_optimizer_schedule


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

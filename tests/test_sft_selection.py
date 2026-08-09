from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from akgr.abduction_model.reproduction import (
    epoch_due,
    is_better_validation,
    prune_sft_checkpoints,
    sft_validation_loss,
    write_best_checkpoint_pointer,
)
from akgr.abduction_model.experiment_runner import _require_selected_healthy_checkpoint


def test_epoch_schedule_always_includes_final_epoch():
    assert not epoch_due(1, 10, 2)
    assert epoch_due(2, 10, 2)
    assert epoch_due(9, 9, 5)


def test_stage_specific_best_validation_selection():
    assert not is_better_validation(
        "conditional",
        {"validation_loss": 0.1, "condition_accuracy": 1.0, "health_pass": False},
        None,
    )
    unconditional = {"validation_loss": 2.0, "jaccard": 0.2}
    assert is_better_validation(
        "unconditional", {"validation_loss": 1.5, "jaccard": 0.0}, unconditional
    )
    assert not is_better_validation(
        "unconditional", {"validation_loss": 2.1, "jaccard": 1.0}, unconditional
    )

    conditional = {
        "validation_loss": 2.0,
        "condition_accuracy": 0.5,
        "jaccard": 0.2,
    }
    assert is_better_validation(
        "conditional",
        {"validation_loss": 3.0, "condition_accuracy": 0.6, "jaccard": 0.0},
        conditional,
    )
    assert is_better_validation(
        "conditional",
        {"validation_loss": 3.0, "condition_accuracy": 0.5, "jaccard": 0.3},
        conditional,
    )
    assert not is_better_validation(
        "conditional",
        {"validation_loss": 1.0, "condition_accuracy": 0.4, "jaccard": 1.0},
        conditional,
    )


def test_best_pointer_and_retention_keep_best_plus_latest(tmp_path):
    for epoch in (2, 5, 6, 10):
        (tmp_path / f"unconditional-epoch-{epoch}").mkdir()
    (tmp_path / "conditional-epoch-5").mkdir()

    record = {
        "stage": "unconditional",
        "stage_epoch": 2,
        "global_step": 3,
        "validation_loss": 1.0,
        "jaccard": 0.25,
    }
    pointer = write_best_checkpoint_pointer(
        tmp_path / "unconditional-best.json",
        checkpoint=tmp_path / "unconditional-epoch-2",
        record=record,
    )
    payload = json.loads(pointer.read_text())
    assert payload["checkpoint"] == "unconditional-epoch-2"
    assert payload["selection"] == ["validation_loss:min", "jaccard:max"]
    assert (tmp_path / "unconditional-best").resolve() == (
        tmp_path / "unconditional-epoch-2"
    ).resolve()

    removed = prune_sft_checkpoints(
        tmp_path, stage="unconditional", keep_last=2, best_epoch=2
    )
    assert {path.name for path in removed} == {"unconditional-epoch-5"}
    assert {path.name for path in tmp_path.glob("unconditional-epoch-*")} == {
        "unconditional-epoch-2",
        "unconditional-epoch-6",
        "unconditional-epoch-10",
    }
    assert (tmp_path / "conditional-epoch-5").is_dir()


def test_stage_transition_requires_selected_checkpoint_that_passed_health_gate(tmp_path):
    selected = tmp_path / "unconditional-epoch-10"
    selected.mkdir()
    other = tmp_path / "unconditional-epoch-20"
    other.mkdir()
    pointer = tmp_path / "unconditional-best.json"
    pointer.write_text(json.dumps({
        "checkpoint": selected.name,
        "validation": {"health_pass": True, "parse_ok": 0.95, "eos_rate": 1.0},
    }))

    assert _require_selected_healthy_checkpoint(
        selected, stage="unconditional"
    ) == selected.resolve()
    with pytest.raises(ValueError, match="requires selected checkpoint"):
        _require_selected_healthy_checkpoint(other, stage="unconditional")

    pointer.write_text(json.dumps({
        "checkpoint": selected.name,
        "validation": {"health_pass": False},
    }))
    with pytest.raises(ValueError, match="did not pass"):
        _require_selected_healthy_checkpoint(selected, stage="unconditional")


class _LossModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, *, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=input_ids[0, 0].float())


def test_validation_loss_is_weighted_by_batch_size():
    batches = [
        SimpleNamespace(
            input_ids=torch.tensor([[2], [2]]),
            attention_mask=torch.ones(2, 1),
            labels=torch.ones(2, 1, dtype=torch.long),
        ),
        SimpleNamespace(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.ones(1, 1),
            labels=torch.ones(1, 1, dtype=torch.long),
        ),
    ]
    model = _LossModel()
    loss = sft_validation_loss(model=model, dataloader=batches, prepare=lambda batch: batch)
    assert model.eval_called
    assert loss == pytest.approx(3.0)

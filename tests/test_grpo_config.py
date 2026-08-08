import pytest

pytest.importorskip("smatch")
pytest.importorskip("trl")

import akgr.abduction_model.reproduction as reproduction
from akgr.abduction_model.reproduction import build_grpo_config


def test_grpo_config_pins_paper_group_and_disables_external_reporting(tmp_path):
    config = {
        "experiment": {"seed": 42},
        "grpo": {
            "epochs": 1, "learning_rate": 1e-5, "beta": 0.1, "epsilon": 0.2,
            "per_device_train_batch_size": 4, "max_completion_length": 8,
            "reward_weights": {"jaccard": 1.0, "dice": 0.5, "overlap": 0.5, "condition": 1.0},
        },
    }
    result = build_grpo_config(config, tmp_path, max_steps=1)
    assert result.num_generations == 4
    assert result.per_device_train_batch_size == 4
    assert result.reward_weights == [1.0, 0.5, 0.5, 1.0]
    assert result.eval_strategy.value == "no"
    assert result.report_to == []
    assert result.max_steps == 1


def test_named_reward_functions_return_python_floats_and_persist_means(monkeypatch, tmp_path):
    def fake_scoring(**kwargs):
        method = kwargs["scoring_method"][0]
        values = {
            "jaccard": {"jaccard": 0.8},
            "dice": {"dice": 0.7},
            "overlap": {"overlap": 0.6},
            "validity": {"validity": 1.0, "enumber": 1.0, "pnumber": 1.0},
        }
        return [values[method]], []

    monkeypatch.setattr(reproduction, "scoring_input_act_batch_condition", fake_scoring)
    log_path = reproduction.configure_reproduction_logging(tmp_path / "reproduction.log")
    rewards = reproduction.make_grpo_reward_functions(
        condition="pattern", graph_samplers={}, searching_split="train"
    )
    assert [fn.__name__ for fn in rewards] == [
        "jaccard_reward", "dice_reward", "overlap_reward", "condition_reward"
    ]
    kwargs = dict(prompts=["1"], completions=["-1 1"], source=["1"], target=["-1 1"], condition=["p e"])
    assert [fn(**kwargs) for fn in rewards] == [[0.8], [0.7], [0.6], [1.0]]
    assert all(isinstance(fn(**kwargs)[0], float) for fn in rewards)
    for handler in reproduction.LOGGER.handlers:
        handler.flush()
    log_text = log_path.read_text(encoding="utf-8")
    assert "GRPO reward component jaccard mean=0.800000" in log_text
    assert "GRPO reward component condition mean=1.000000" in log_text
    assert "GRPO combined reward mean=2.450000" in log_text

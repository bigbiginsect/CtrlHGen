import pytest
from types import SimpleNamespace

pytest.importorskip("smatch")
trl = pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import akgr.abduction_model.reproduction as reproduction
from akgr.abduction_model.reproduction import build_grpo_config
from akgr.tokenizer import create_reproduction_tokenizer


def _config(batch_size=4):
    raw = {
        "experiment": {"seed": 42},
        "grpo": {
            "epochs": 1, "learning_rate": 1e-5, "beta": 0.1, "epsilon": 0.2,
            "per_device_train_batch_size": batch_size, "max_completion_length": 8,
            "save_steps": 10, "save_total_limit": 2,
            "reward_weights": {"jaccard": 1.0, "dice": 0.5, "overlap": 0.5, "condition": 1.0},
        },
    }
    return SimpleNamespace(raw=raw, seed=42, condition="pattern")


def test_grpo_config_pins_paper_group_and_disables_external_reporting(tmp_path):
    config = _config()
    result = build_grpo_config(config, tmp_path, max_steps=1)
    assert result.num_generations == 4
    assert result.per_device_train_batch_size == 4
    assert result.reward_weights == [1.0, 0.5, 0.5, 1.0]
    assert result.eval_strategy.value == "no"
    assert result.report_to == []
    assert result.max_steps == 1
    assert result.save_steps == 10
    assert result.save_total_limit == 2
    assert result.optim.value == "adamw_torch"
    assert result.lr_scheduler_type.value == "linear"
    assert result.warmup_steps == 0
    assert result.num_iterations == 1
    assert result.scale_rewards is True
    assert result.gradient_accumulation_steps == 1
    assert result.max_prompt_length == 512
    assert result.temperature == 0.9
    assert result.top_p == 1.0
    assert result.top_k == 50
    assert result.repetition_penalty == 1.0
    assert result.use_vllm is False
    assert result.bf16 is False
    assert result.fp16 is False
    assert result.gradient_checkpointing is False


def test_create_grpo_trainer_disables_policy_and_reference_dropout(tmp_path):
    tokenizer = create_reproduction_tokenizer(4, 2)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_embd=24, n_head=4,
        n_positions=32, n_ctx=32,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    dataset = datasets.Dataset.from_dict({
        "prompt": ["1", "1", "1", "1"],
        "condition": ["p e", "p e", "p e", "p e"],
    })
    trainer = reproduction.create_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=_config(),
        output_dir=tmp_path,
        graph_samplers={},
        max_steps=1,
    )
    assert all(module.p == 0.0 for module in trainer.model.modules() if isinstance(module, torch.nn.Dropout))
    assert all(module.p == 0.0 for module in trainer.ref_model.modules() if isinstance(module, torch.nn.Dropout))


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

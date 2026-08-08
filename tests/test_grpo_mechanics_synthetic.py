import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")
trl = pytest.importorskip("trl")

from akgr.tokenizer import create_reproduction_tokenizer


def _model(tokenizer):
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_embd=24, n_head=4,
        n_positions=32, n_ctx=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))


def _reward(completions, **kwargs):
    # GRPO normalizes rewards within each generation group.  A value derived
    # from the group position guarantees non-zero advantages even when the
    # randomly initialized model emits identical completions.
    return [float(index % 4) for index, _ in enumerate(completions)]


def _args(output_dir, max_steps):
    return trl.GRPOConfig(
        output_dir=str(output_dir),
        use_cpu=True,
        seed=17,
        max_steps=max_steps,
        learning_rate=1e-4,
        num_generations=4,
        per_device_train_batch_size=4,
        max_prompt_length=8,
        max_completion_length=3,
        beta=0.0,
        remove_unused_columns=False,
        eval_strategy="no",
        report_to=[],
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
    )


@pytest.mark.synthetic
def test_grpo_cpu_one_step_and_resume_to_two(tmp_path):
    tokenizer = create_reproduction_tokenizer(4, 2)
    tokenizer.padding_side = "left"
    dataset = datasets.Dataset.from_dict({"prompt": ["1", "2", "3", "4"]})
    first_model = _model(tokenizer)
    before = {
        name: parameter.detach().clone()
        for name, parameter in first_model.named_parameters()
    }
    first = trl.GRPOTrainer(
        model=first_model, reward_funcs=_reward,
        args=_args(tmp_path, 1), train_dataset=dataset,
        processing_class=tokenizer,
    )
    first.train()
    checkpoint = tmp_path / "checkpoint-1"
    assert first.state.global_step == 1
    assert (checkpoint / "trainer_state.json").is_file()
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in first_model.named_parameters()
    )

    second = trl.GRPOTrainer(
        model=_model(tokenizer), reward_funcs=_reward,
        args=_args(tmp_path, 2), train_dataset=dataset,
        processing_class=tokenizer,
    )
    second.train(resume_from_checkpoint=str(checkpoint))
    assert second.state.global_step == 2
    assert (tmp_path / "checkpoint-2" / "trainer_state.json").is_file()

"""Read-only SFT diagnostics for prompt, teacher-forcing, and greedy decoding."""

from __future__ import annotations

import argparse
import json

import torch

from akgr.abduction_model.experiment_runner import (
    _datasets,
    _file_sha256,
    _loader,
    _prompt_length,
    _require_cuda,
)
from akgr.evaluation import parse_action_status
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.seed import seed_everything
from akgr.tokenizer import prepare_batch
from akgr.utils.load_util import load_reproduction_checkpoint


def diagnose(config, checkpoint, *, split: str, batch_size: int, max_samples: int,
             prompt_condition: str) -> dict:
    loaded = load_reproduction_checkpoint(
        checkpoint,
        mode="test",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    checkpoint_stage = loaded.metadata["stage"]
    train_variant = (
        config.raw["training"][checkpoint_stage]["data_variant"]
        if checkpoint_stage in {"unconditional", "conditional"}
        else config.raw["grpo"]["data_variant"]
    )
    datasets, _, _ = _datasets(config, [split], train_variant=train_variant)
    dataset = datasets[split].select(range(min(int(max_samples), len(datasets[split]))))
    dataloader = _loader(dataset, batch_size, config.seed, False)
    condition = (
        loaded.metadata["condition"]
        if prompt_condition == "checkpoint"
        else "unconditional"
    )
    delimiter = config.raw["tokenizer"]["condition_delimiter"]
    device = _require_cuda("SFT diagnostics")
    model = loaded.model.to(device).eval()
    tokenizer = loaded.tokenizer

    token_correct = token_total = 0
    sequence_correct = sequence_total = 0
    eos_correct = eos_total = 0
    first_correct = first_total = 0
    prompt_first_correct = prompt_first_total = 0
    prompt_teacher_agree = 0
    prompt_contract_equal = True
    greedy_eos = greedy_parse = greedy_total = 0
    greedy_lengths = []
    weighted_loss = 0.0
    examples = []

    with torch.no_grad():
        for sample in dataloader:
            common = dict(
                device=device,
                sample=sample,
                tokenizer=tokenizer,
                is_gpt=True,
                src_len=_prompt_length(config),
                tgt_len=config.raw["generation"]["max_new_tokens"],
                condition=condition,
                condition_delimiter=delimiter,
            )
            train_batch = prepare_batch(is_gen=False, **common)
            prompt_batch = prepare_batch(is_gen=True, **common)
            batch_count = len(train_batch.source)

            for row in range(batch_count):
                train_prompt = train_batch.input_ids[row][
                    train_batch.source_attention_mask[row].bool()
                ].tolist()
                generation_prompt = prompt_batch.input_ids[row][
                    prompt_batch.attention_mask[row].bool()
                ].tolist()
                prompt_contract_equal &= train_prompt == generation_prompt

            output = model(
                input_ids=train_batch.input_ids,
                attention_mask=train_batch.attention_mask,
                labels=train_batch.labels,
            )
            weighted_loss += float(output.loss) * batch_count
            shifted_labels = train_batch.labels[:, 1:]
            shifted_predictions = output.logits[:, :-1].argmax(dim=-1)
            active = shifted_labels != -100
            matches = shifted_predictions == shifted_labels
            token_correct += int((matches & active).sum())
            token_total += int(active.sum())
            sequence_correct += int(((matches | ~active).all(dim=1)).sum())
            sequence_total += batch_count
            eos_mask = shifted_labels == tokenizer.eos_token_id
            eos_correct += int((matches & eos_mask).sum())
            eos_total += int(eos_mask.sum())

            position_ids = prompt_batch.attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(prompt_batch.attention_mask == 0, 0)
            prompt_logits = model(
                input_ids=prompt_batch.input_ids,
                attention_mask=prompt_batch.attention_mask,
                position_ids=position_ids,
            ).logits[:, -1]
            prompt_predictions = prompt_logits.argmax(dim=-1)

            first_targets = []
            teacher_first_predictions = []
            for row in range(batch_count):
                label_positions = torch.nonzero(
                    train_batch.labels[row] != -100, as_tuple=False
                ).flatten()
                first_position = int(label_positions[0])
                first_target = int(train_batch.labels[row, first_position])
                teacher_prediction = int(output.logits[row, first_position - 1].argmax())
                first_targets.append(first_target)
                teacher_first_predictions.append(teacher_prediction)
            first_targets_tensor = torch.tensor(first_targets, device=device)
            teacher_first_tensor = torch.tensor(teacher_first_predictions, device=device)
            first_correct += int((teacher_first_tensor == first_targets_tensor).sum())
            first_total += batch_count
            prompt_first_correct += int((prompt_predictions == first_targets_tensor).sum())
            prompt_first_total += batch_count
            prompt_teacher_agree += int((prompt_predictions == teacher_first_tensor).sum())

            generated = model.generate(
                input_ids=prompt_batch.input_ids,
                attention_mask=prompt_batch.attention_mask,
                do_sample=False,
                max_new_tokens=int(config.raw["generation"]["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[:, prompt_batch.input_ids.shape[1]:]
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, prediction in enumerate(predictions):
                token_ids = generated[row].tolist()
                emitted_eos = tokenizer.eos_token_id in token_ids
                length = token_ids.index(tokenizer.eos_token_id) + 1 if emitted_eos else len(token_ids)
                parse_ok, _ = parse_action_status(prediction)
                greedy_eos += int(emitted_eos)
                greedy_parse += int(parse_ok)
                greedy_total += 1
                greedy_lengths.append(length)
                if len(examples) < 5:
                    examples.append({
                        "source": train_batch.source[row],
                        "target": train_batch.target[row],
                        "prompt": tokenizer.decode(
                            prompt_batch.input_ids[row][prompt_batch.attention_mask[row].bool()],
                            skip_special_tokens=False,
                        ),
                        "first_target": tokenizer.convert_ids_to_tokens(first_targets[row]),
                        "teacher_first": tokenizer.convert_ids_to_tokens(
                            teacher_first_predictions[row]
                        ),
                        "prompt_first": tokenizer.convert_ids_to_tokens(
                            int(prompt_predictions[row])
                        ),
                        "prediction": prediction,
                    })

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_stage": checkpoint_stage,
        "split": split,
        "prompt_condition": condition,
        "sample_count": sequence_total,
        "prompt_contract_equal": bool(prompt_contract_equal),
        "teacher_forced": {
            "loss": weighted_loss / sequence_total,
            "token_accuracy": token_correct / token_total,
            "sequence_accuracy": sequence_correct / sequence_total,
            "first_token_accuracy": first_correct / first_total,
            "eos_label_count": eos_total,
            "eos_accuracy": eos_correct / eos_total if eos_total else None,
        },
        "prompt_only": {
            "first_token_accuracy": prompt_first_correct / prompt_first_total,
            "agreement_with_teacher_forced": prompt_teacher_agree / prompt_first_total,
        },
        "greedy": {
            "eos_rate": greedy_eos / greedy_total,
            "parse_ok": greedy_parse / greedy_total,
            "mean_generated_tokens": sum(greedy_lengths) / greedy_total,
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--prompt-condition", choices=["checkpoint", "unconditional"], default="checkpoint"
    )
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    seed_everything(config.seed)
    result = diagnose(
        config,
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        prompt_condition=args.prompt_condition,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

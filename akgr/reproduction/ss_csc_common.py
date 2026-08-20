"""Shared verified artifacts and likelihood utilities for SS-CSC."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from akgr.reproduction.contracts import ConditionSpec
from akgr.tokenizer import build_prompt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_pair_artifact(summary_path: Path, name: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    summary_path = summary_path.expanduser().resolve()
    summary = read_json(summary_path)
    if summary.get("kind") != "ss_csc_full_pair_data_audit" or summary.get("status") != "passed":
        raise ValueError("Pair summary is not a passed SS-CSC Experiment A artifact")
    if summary.get("data_gate", {}).get("passed") is not True:
        raise ValueError("SS-CSC Experiment A data gate did not pass")
    artifact = summary.get("artifacts", {}).get(name)
    if not isinstance(artifact, dict):
        raise ValueError(f"Pair summary has no {name!r} artifact")
    path = (summary_path.parent / str(artifact["path"])).resolve()
    if path.parent != summary_path.parent or not path.is_file():
        raise ValueError(f"Invalid SS-CSC artifact path: {path}")
    if sha256_file(path) != artifact.get("sha256"):
        raise ValueError(f"SS-CSC artifact hash mismatch: {path}")
    rows = read_jsonl(path)
    if len(rows) != int(artifact.get("count", -1)):
        raise ValueError(f"SS-CSC artifact count mismatch: {path}")
    return summary, rows, path


def condition_prompt(source: str, condition: str, tokenizer, delimiter: str) -> str:
    return build_prompt(
        source,
        ConditionSpec("specific_relation", str(condition)),
        tokenizer,
        condition_delimiter=delimiter,
    )


def reference_mean_log_probs(
    *, model, tokenizer, prompts: Sequence[str], targets: Sequence[str], device,
    max_length: int,
) -> torch.Tensor:
    """Return differentiable per-target token-mean log probabilities."""
    if len(prompts) != len(targets) or not prompts:
        raise ValueError("Likelihood batches require equally sized non-empty prompts/targets")
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        encoded = tokenizer(
            list(prompts), list(targets), padding="longest",
            truncation=True, max_length=int(max_length), return_tensors="pt",
        ).to(device)
        prompt_mask = tokenizer(
            list(prompts), padding="max_length", truncation=True,
            max_length=encoded.input_ids.shape[-1], return_tensors="pt",
        ).attention_mask.to(device)
        labels = encoded.input_ids.clone()
        labels[prompt_mask == 1] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        logits = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        losses = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1),
            reduction="none", ignore_index=-100,
        ).view(shifted_labels.shape)
        counts = shifted_labels.ne(-100).sum(dim=1)
        if torch.any(counts == 0):
            raise RuntimeError("Likelihood batch contains an empty target")
        return -losses.sum(dim=1) / counts
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()


def chunks(rows: Sequence[Any], size: int):
    for start in range(0, len(rows), int(size)):
        yield rows[start:start + int(size)]


def pair_likelihood_rows(
    *, model, tokenizer, pairs: Sequence[Mapping[str, Any]], device,
    delimiter: str, max_length: int, batch_size: int, margin: float,
) -> list[dict[str, Any]]:
    model.eval()
    output = []
    with torch.no_grad():
        for batch in chunks(list(pairs), batch_size):
            prompts, targets = [], []
            for pair in batch:
                source = str(pair["source"])
                c1, c2 = pair["left"]["condition_value"], pair["right"]["condition_value"]
                h1, h2 = pair["left"]["target"], pair["right"]["target"]
                prompts.extend([
                    condition_prompt(source, c1, tokenizer, delimiter),
                    condition_prompt(source, c2, tokenizer, delimiter),
                    condition_prompt(source, c2, tokenizer, delimiter),
                    condition_prompt(source, c1, tokenizer, delimiter),
                ])
                targets.extend([h1, h2, h1, h2])
            scores = reference_mean_log_probs(
                model=model, tokenizer=tokenizer, prompts=prompts, targets=targets,
                device=device, max_length=max_length,
            ).detach().cpu().tolist()
            for index, pair in enumerate(batch):
                h1c1, h2c2, h1c2, h2c1 = (float(value) for value in scores[4 * index:4 * index + 4])
                margin1, margin2 = h1c1 - h1c2, h2c2 - h2c1
                c1_selects_h1 = h1c1 > h2c1
                c2_selects_h2 = h2c2 > h1c2
                output.append({
                    "pair_id": pair["pair_id"],
                    "observation_sha256": pair["observation_sha256"],
                    "h1_c1_mean_logp": h1c1,
                    "h2_c2_mean_logp": h2c2,
                    "h1_c2_swapped_mean_logp": h1c2,
                    "h2_c1_swapped_mean_logp": h2c1,
                    "h1_prompt_margin": margin1,
                    "h2_prompt_margin": margin2,
                    "h1_margin_satisfied": margin1 >= float(margin),
                    "h2_margin_satisfied": margin2 >= float(margin),
                    "symmetric_margin_satisfied": margin1 >= float(margin) and margin2 >= float(margin),
                    "c1_selects_h1": c1_selects_h1,
                    "c2_selects_h2": c2_selects_h2,
                    "condition_consistent_selection_accuracy": (
                        float(c1_selects_h1) + float(c2_selects_h2)
                    ) / 2.0,
                    "bilateral_condition_consistent_selection": c1_selects_h1 and c2_selects_h2,
                })
    return output


def mean(values: Sequence[float]) -> float | None:
    return math.fsum(float(value) for value in values) / len(values) if values else None


def summarize_pair_likelihood(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "assigned_target_mean_logp": mean([
            (float(row["h1_c1_mean_logp"]) + float(row["h2_c2_mean_logp"])) / 2.0
            for row in rows
        ]),
        "swapped_target_mean_logp": mean([
            (float(row["h1_c2_swapped_mean_logp"]) + float(row["h2_c1_swapped_mean_logp"])) / 2.0
            for row in rows
        ]),
        "mean_prompt_margin": mean([
            (float(row["h1_prompt_margin"]) + float(row["h2_prompt_margin"])) / 2.0
            for row in rows
        ]),
        "directional_margin_satisfaction_rate": mean([
            (float(row["h1_margin_satisfied"]) + float(row["h2_margin_satisfied"])) / 2.0
            for row in rows
        ]),
        "symmetric_margin_satisfaction_rate": mean([
            float(row["symmetric_margin_satisfied"]) for row in rows
        ]),
        "condition_consistent_selection_accuracy": mean([
            float(row["condition_consistent_selection_accuracy"]) for row in rows
        ]),
        "bilateral_condition_consistent_selection_rate": mean([
            float(row["bilateral_condition_consistent_selection"]) for row in rows
        ]),
    }

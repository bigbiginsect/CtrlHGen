from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("tokenizers")

from akgr.reproduction.sc_idc_phase2_data import (
    _condition_manifest,
    build_condition_rows,
    create_parent_import_contract,
    load_frozen_condition_manifest,
    require_shared_rl_manifest_hash,
    verify_parent_import_contract,
)
from akgr.tokenizer import (
    create_reproduction_tokenizer,
    dynamic_condition_value_from_target,
    prepare_batch,
    unique_condition_values_from_target,
)
from akgr.utils.load_util import save_reproduction_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dynamic_sampler_is_uniform_over_unique_values_not_occurrences():
    target = "i -1 1 u -1 2 -2 3"
    assert unique_condition_values_from_target("specific_relation", target) == ("-1", "-2")
    observed = {
        dynamic_condition_value_from_target(
            "specific_relation", target, seed=42, record_id="record-a", epoch=epoch
        )
        for epoch in range(64)
    }
    assert observed == {"-1", "-2"}
    assert dynamic_condition_value_from_target(
        "specific_relation", target, seed=42, record_id="record-a", epoch=7
    ) == dynamic_condition_value_from_target(
        "specific_relation", target, seed=42, record_id="record-a", epoch=7
    )


def test_prepare_batch_supports_dynamic_train_and_frozen_validation_values():
    tokenizer = create_reproduction_tokenizer(8, 8)
    sample = {
        "source": ["1 2"],
        "target": ["i -1 1 -2 2"],
        "pattern_id": [0],
        "record_id": ["record-a"],
    }
    dynamic = prepare_batch(
        "cpu", sample, tokenizer, True, 32, 16, False, "specific_relation",
        condition_delimiter="COND", condition_seed=42, condition_epoch=3,
    )
    assert dynamic.conditions[0].value in {"-1", "-2"}
    frozen = prepare_batch(
        "cpu", sample, tokenizer, True, 32, 16, False, "specific_relation",
        condition_delimiter="COND", condition_value_by_record_id={"record-a": "-2"},
    )
    assert frozen.conditions[0].value == "-2"
    with pytest.raises(ValueError, match="missing record IDs"):
        prepare_batch(
            "cpu", sample, tokenizer, True, 32, 16, False, "specific_relation",
            condition_delimiter="COND", condition_value_by_record_id={},
        )


def test_frozen_rows_are_byte_stable_and_repeated_value_has_one_control_row():
    record = {
        "record_id": "r1",
        "answers": [0],
        "query": ["(", "i", "(", "p", "(", 0, ")", "(", "e", "(", 0, ")", ")", ")",
                  "(", "p", "(", 0, ")", "(", "e", "(", 1, ")", ")", ")", ")"],
        "pattern_str": "(i,(p,(e)),(p,(e)))",
    }
    first = build_condition_rows([record], seed=42, purpose="validation")
    second = build_condition_rows([record], seed=42, purpose="validation")
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert len(first) == 1
    assert first[0]["condition_value"] == "-1"
    assert first[0]["topology"] == "repeated"


def test_sealed_final_conditions_reject_sft_consumer(tmp_path):
    config = SimpleNamespace(
        semantic_hash="target", data_hash="data", kg_hash="kg", seed=42
    )
    rows = [{
        "record_id": "r", "condition_kind": "specific_relation",
        "condition_value": "-1", "selection_seed": 1, "topology": "first_only",
        "pattern_str": "p", "query_sha256": "q", "supervision_sha256": "s",
        "target_sha256": "t",
    }]
    reference = _condition_manifest(
        output_dir=tmp_path, purpose="final_evaluation", rows=rows,
        target_config=config, source_reference={"kind": "test"},
    )
    with pytest.raises(PermissionError, match="cannot load"):
        load_frozen_condition_manifest(
            tmp_path / reference["manifest_path"], target_config=config,
            purpose="final_evaluation", consumer="conditional_sft",
        )


def test_paired_grpo_requires_identical_rl_manifest_hash():
    first = {"conditions": {"rl_train": {"manifest_sha256": "same"}}}
    second = deepcopy(first)
    assert require_shared_rl_manifest_hash(first, second) == "same"
    second["conditions"]["rl_train"]["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="must share"):
        require_shared_rl_manifest_hash(first, second)


def _synthetic_config(
    path: Path, *, condition: str, semantic_hash: str, sampling_manifest: Path,
    raw: dict, data_hash: str = "data", kg_hash: str = "kg",
):
    path.write_text(json.dumps(raw), encoding="utf-8")
    return SimpleNamespace(
        source_path=path.resolve(), raw=raw, condition=condition,
        semantic_hash=semantic_hash, data_hash=data_hash, kg_hash=kg_hash,
        sampling_manifest_path=sampling_manifest.resolve(),
    )


def test_cross_condition_parent_import_is_explicit_and_tamper_evident(tmp_path):
    sampling_manifest = tmp_path / "sampling-manifest.json"
    sampling_manifest.write_text(json.dumps({
        "stats": {"nentity": 8, "nrelation": 8}
    }), encoding="utf-8")
    raw_model = {
        "type": "gpt2", "initialization": "random", "n_layer": 1,
        "n_embd": 32, "n_head": 4, "n_positions": 64, "n_ctx": 64,
        "tie_word_embeddings": True,
    }
    source_raw = {"model": raw_model, "tokenizer": {"condition_delimiter": "COND"}}
    target_raw = deepcopy(source_raw)
    source = _synthetic_config(
        tmp_path / "source.json", condition="pattern", semantic_hash="source-hash",
        sampling_manifest=sampling_manifest, raw=source_raw,
    )
    target = _synthetic_config(
        tmp_path / "target.json", condition="specific_relation", semantic_hash="target-hash",
        sampling_manifest=sampling_manifest, raw=target_raw,
    )
    tokenizer = create_reproduction_tokenizer(8, 8)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_embd=32, n_head=4,
        n_positions=64, n_ctx=64, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "unconditional-epoch-1"
    save_reproduction_checkpoint(
        checkpoint, model=model, tokenizer=tokenizer, stage="unconditional",
        stage_epoch=1, global_step=1, condition="unconditional",
        experiment_config=source_raw, experiment_config_hash=source.semantic_hash,
        seed=42, optimizer=optimizer, scheduler=scheduler,
        data_manifest_hash=_sha256(sampling_manifest),
    )
    (tmp_path / "unconditional-best.json").write_text(json.dumps({
        "stage": "unconditional", "checkpoint": checkpoint.name,
        "selection": ["parse_ok:max"], "validation": {"health_pass": True},
    }), encoding="utf-8")

    contract = create_parent_import_contract(
        source_config=source, target_config=target, checkpoint=checkpoint,
        command=["synthetic"],
    )
    assert contract["transition"] == {
        "weights_only": True, "optimizer_reset": True,
        "scheduler_reset": True, "rng_not_restored": True,
    }
    verify_parent_import_contract(contract, target_config=target, checkpoint=checkpoint)
    tampered_target = _synthetic_config(
        tmp_path / "tampered.json", condition="specific_relation",
        semantic_hash="tampered-target", sampling_manifest=sampling_manifest,
        raw=target_raw,
    )
    with pytest.raises(ValueError, match="target config_semantic_hash"):
        verify_parent_import_contract(
            contract, target_config=tampered_target, checkpoint=checkpoint
        )
    (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tree hash mismatch"):
        verify_parent_import_contract(contract, target_config=target, checkpoint=checkpoint)

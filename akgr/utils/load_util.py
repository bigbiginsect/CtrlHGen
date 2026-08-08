import yaml
def load_yaml(filename: str):
    with open(filename, 'r') as f:
        obj = yaml.safe_load(f)
    return obj

import pandas as pd
def load_jsonl(data_path):
    """
    return a list of dict
    """
    data_dict = pd.read_json(data_path,
        orient='records', lines=True).to_dict(orient='records')
    return data_dict

def load_csv(data_path, **kwargs):
    df = pd.read_csv(data_path, **kwargs)
    return df

def load_and_filter_query_patterns(
        file_name,
        max_dep=3, exclu='', column='original', with_depth=False):
    """
    Terminology:
    - pattern id: the number of the pattern
    - pattern str: the parentheses expression
    - pattern abbr: the abbreviation

    Input:
    - file_name = name of the csv file
    - max_dep = maximum depth
    - exclu = string consists of 'u', 'n', 'i', 'p', seperated by '|'. operations to exclude
    - column = default 'original'

    Output:
    - pattern_filtered
    """
    pattern_table = pd.read_csv(file_name, index_col='id')#.reset_index(drop = True)
    pattern_filterd = pattern_table[[column, 'pattern_abbr', 'original_depth']]

    pattern_filterd = pattern_filterd.rename({column: 'pattern_str'}, axis='columns')

    pattern_filterd = pattern_filterd.loc[pattern_filterd.original_depth <= max_dep]
    # exclude_pattern = '|'.join(excep)
    if exclu is not None:
        pattern_filterd = pattern_filterd.loc[~pattern_filterd.pattern_str.str.contains(exclu, case=False)]
    # pattern_filtered = {}
    # for i in range(pattern_table.shape[0]):
    #     is_selected = True

    #     dep = pattern_table.original_depth[i]
    #     is_selected = dep <= max_dep

    #     if column == 'original':
    #         pattern_str = pattern_table.original[i]
    #     for ch in excep:
    #         if ch in pattern_str:
    #             is_selected = False
    #     if not is_selected: continue
    #     pattern_id = int(pattern_table.formula_id[i][-4:])
    #     if 'Abbreviation' in pattern_table.columns:
    #         abbr = pattern_table.Abbreviation[i]
    #     else:
    #         abbr = ''
    #     if with_depth == True:
    #         pattern_filtered[pattern_id] = (pattern_str, abbr, dep)
    #     else:
    #         pattern_filtered[pattern_id] = (pattern_str, abbr)

    return pattern_filterd

import os
import pickle

def load_sampled_dataset(data_root, dataname, scale, answer_size,
                         splits=['train', 'valid', 'test'],
                         method='jsonl'):
    """
    Output: data_dict ["train"/"valid"/"test"]
    """
    data_dict = {}
    for split in splits:
        data_path = os.path.join(data_root, dataname, \
            f'{dataname}-{scale}-{answer_size}-{split}-a2q.{method}')
        if method == 'jsonl':
            data_dict[split] = load_jsonl(data_path)
        elif method == 'pkl':
            with open(data_path, 'rb') as f:
                data_dict[split] = pickle.load(f)

    stats_path = os.path.join(data_root, dataname, f'stats.txt')
    with open(stats_path) as f:
        lines = f.readlines()
        nentity = int(lines[0].split('\t')[-1])
        nrelation = int(lines[1].split('\t')[-1])
    # data_dict['test'] = [
    #     # {"answers":[3454,6345,3018,20909,19824,2802,9397,5144,16510,23708,23678],"query":["(","i","(","n","(","p","(",-549,")","(","e","(",12994,")",")",")",")","(","p","(",-547,")","(","e","(",2618,")",")",")",")"],"pattern_str":"(i,(n,(p,(e))),(p,(e)))"},
    #     # {"answers":[8645,6511,11929,9818,21918,21471],"query":["(","p","(",-195,")","(","u","(","p","(",-269,")","(","e","(",9116,")",")",")","(","p","(",-194,")","(","e","(",9818,")",")",")",")",")"],"pattern_str":"(p,(u,(p,(e)),(p,(e))))"},
    #     {"answers":[18369,22403,23044,19272,5898,1932,6160,1553,5653,15063,16672,12579,1060,14438,1511,23273,15917,15918,1069,15533,15924,13495,15929,24314,24317,7615],"query":["(","i","(","i","(","n","(","p","(",-659,")","(","e","(",18646,")",")",")",")","(","p","(",-659,")","(","e","(",7020,")",")",")",")","(","p","(",-659,")","(","e","(",7020,")",")",")",")"],"pattern_str":"(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))"},
    #     # {"answers":[9280,4355,12101,11334,23182,8914,1108,8024,10908,9954,23590,11688,7275,23212,2732,9135,17584,9140],"query":["(","i","(","n","(","p","(",-202,")","(","p","(",0,")","(","e","(",1949,")",")",")",")",")","(","p","(",-567,")","(","e","(",8134,")",")",")",")"],"pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"},
    #     {"answers":[17410,13187,3781,4581,4968,15085,23121,1845,6870,19801,15068,20029,6527],"query":["(","p","(",-119,")","(","e","(",2916,")",")",")"],"pattern_str":"(p,(e))"}
    # ]
    return data_dict, nentity, nrelation
def jsonl_2_pickle(data_root, dataname, scale, answer_size,
                    splits=['train', 'valid', 'test']):
    data_dict = {}
    for split in splits:
        jsonl_path = os.path.join(data_root, dataname, \
            f'{dataname}-{scale}-{answer_size}-{split}-a2q.jsonl')
        data = load_jsonl(jsonl_path)
        pickle_path = jsonl_path.removesuffix('.jsonl') + '.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)

import torch
import torch.nn as nn
import transformers
from dataclasses import dataclass
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
def load_model(path, contents:str, epoch,
               return_huggingface_model:bool,
               model=None, optimizer=None, scheduler=None):
    """
    contents:
        - "state_dicts": weights only, need to instantiate first
        - "model": full model, load directly
    """
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    if contents == 'state_dicts':
        print(f'# Loading checkpoint (state_dicts) {path}')
        if model == None or optimizer == None or scheduler == None:
            print('# Error: need to instantiate and pass model, optimizer, scheduler')
            exit()
        checkpoint = torch.load(path,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    elif contents == 'model':
        print(f'# Loading checkpoint (model) {path}')
        if model is not None or optimizer is not None or scheduler is not None:
            print('# Error: cannot pass model, optimizer, scheduler with contents="model"')
            exit()
        checkpoint = torch.load(path,weights_only=False)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    elif contents == 'rlmodel':
        print(f'# Loading checkpoint (model) {path}')
        if model is not None or optimizer is not None or scheduler is not None:
            print('# Error: cannot pass model, optimizer, scheduler with contents="model"')
            exit()
        checkpoint = torch.load(path,weights_only=False)
        model = checkpoint['model']
        model.warnings_issued = {}  # 重新添加属性
        def dummy_add_model_tags(self, tags):
            pass
        model.add_model_tags = dummy_add_model_tags.__get__(model)  # 重新绑定方法
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        print(f'# Error: contents "{contents}" not supported')
        exit()
    last_epoch = checkpoint['epoch']
    if 'loss_log' in checkpoint.keys():
        loss_log = checkpoint['loss_log']
    else:
        loss_log = {'train': {}, 'valid': {}}
    if return_huggingface_model and not isinstance(model, transformers.PreTrainedModel):
        print('Yes, returning .transformer')
        model = model.transformer
    return model, optimizer, scheduler, last_epoch, loss_log

import pathlib
def save_model(path, contents:str,
               model, optimizer=None, scheduler=None, epoch=None, loss_log=None):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if contents == 'state_dicts':
        print(f'# Saving checkpoint (state_dicts) {path}')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss_log': loss_log
        }, path)
    elif contents == 'model':
        print(f'# Saving checkpoint (model) {path}')
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': epoch,
            'loss_log': loss_log
        }, path)
    else:
        print(f'# Error: contents "{contents}" not supported')
        exit()


CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class LoadedReproductionCheckpoint:
    path: Path
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizerBase
    metadata: dict
    training_state: dict


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _state_dict_cpu(model) -> dict:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _json_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_config_hash(config) -> str:
    payload = config.to_dict()
    for volatile_key in (
        "_name_or_path",
        "_commit_hash",
        "transformers_version",
        "_attn_implementation_autoset",
    ):
        payload.pop(volatile_key, None)
    return _json_sha256(payload)


def save_reproduction_checkpoint(
    path,
    *,
    model,
    tokenizer,
    stage: str,
    stage_epoch: int,
    global_step: int,
    condition: str,
    experiment_config: dict,
    experiment_config_hash: str,
    seed: int,
    parent_checkpoint: str | None = None,
    optimizer=None,
    scheduler=None,
    data_manifest_hash: str | None = None,
) -> Path:
    """Atomically save a portable Phase A checkpoint directory."""
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        model.config.to_json_file(temporary / "config.json")
        model.save_pretrained(temporary / "model", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "tokenizer")
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "stage": stage,
            "stage_epoch": int(stage_epoch),
            "global_step": int(global_step),
            "condition": condition,
            "parent_checkpoint": str(Path(parent_checkpoint).expanduser().resolve()) if parent_checkpoint else None,
            "seed": int(seed),
            "experiment_config_hash": experiment_config_hash,
            "experiment_config": experiment_config,
            "data_manifest_hash": data_manifest_hash,
            "model_config_hash": _model_config_hash(model.config),
            "tokenizer_vocab_hash": _json_sha256(tokenizer.get_vocab()),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        training_state = {
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": _capture_rng_state(),
        }
        torch.save(training_state, temporary / "training_state.pt")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_reproduction_checkpoint(
    path,
    *,
    mode: str,
    model=None,
    optimizer=None,
    scheduler=None,
    expected_stage: str | None = None,
    expected_condition: str | None = None,
    expected_config_hash: str | None = None,
    expected_data_manifest_hash: str | None = None,
    restore_rng: bool = False,
    map_location="cpu",
) -> LoadedReproductionCheckpoint:
    """Load a parent, resume, or evaluation checkpoint with strict checks."""
    if mode not in {"parent", "resume", "test"}:
        raise ValueError("mode must be parent, resume, or test")
    root = Path(path).expanduser().resolve()
    required = {"metadata.json", "config.json", "model", "training_state.pt", "tokenizer"}
    missing = [name for name in sorted(required) if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {root}; missing: {missing}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {metadata.get('format_version')}")
    if expected_stage is not None and metadata["stage"] != expected_stage:
        raise ValueError(f"Checkpoint stage {metadata['stage']!r} != expected {expected_stage!r}")
    if expected_condition is not None and metadata["condition"] != expected_condition:
        raise ValueError(f"Checkpoint condition {metadata['condition']!r} != expected {expected_condition!r}")
    if expected_config_hash is not None and metadata["experiment_config_hash"] != expected_config_hash:
        raise ValueError("Checkpoint experiment config hash does not match")
    if expected_data_manifest_hash is not None and metadata.get("data_manifest_hash") != expected_data_manifest_hash:
        raise ValueError("Checkpoint data manifest hash does not match")

    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    if model is None:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            root / "model", local_files_only=True
        )
    else:
        saved_model = transformers.AutoModelForCausalLM.from_pretrained(
            root / "model", local_files_only=True
        )
        model.load_state_dict(saved_model.state_dict(), strict=True)
        # save_pretrained may add stable serialization fields such as
        # ``architectures``.  Adopt the saved config so an equivalent freshly
        # instantiated model validates exactly like a directly loaded model.
        model.config = saved_model.config
        model.generation_config = saved_model.generation_config
        del saved_model
    training_state = torch.load(root / "training_state.pt", map_location=map_location, weights_only=False)

    if _model_config_hash(model.config) != metadata["model_config_hash"]:
        raise ValueError("Loaded model config does not match checkpoint metadata")
    if _json_sha256(tokenizer.get_vocab()) != metadata["tokenizer_vocab_hash"]:
        raise ValueError("Loaded tokenizer vocab does not match checkpoint metadata")
    if mode == "resume":
        if optimizer is None or scheduler is None:
            raise ValueError("resume mode requires an instantiated optimizer and scheduler")
        if training_state["optimizer_state_dict"] is None or training_state["scheduler_state_dict"] is None:
            raise ValueError("Checkpoint does not contain optimizer/scheduler state")
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
        if restore_rng:
            _restore_rng_state(training_state["rng_state"])
    return LoadedReproductionCheckpoint(root, model, tokenizer, metadata, training_state)

if __name__ == '__main__':
    print(yaml.dump(load_yaml('akgr/configs/config-dataloader.yml')))

#!/usr/bin/python3
import argparse

import os, sys
import hashlib
import json
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader

# sys.path.append('./utils/')
from akgr.utils.load_util import load_yaml, load_csv, load_sampled_dataset
# from akgr.utils.parsing_util import qry_wordlist_2_actions, qry_wordlist_2_actions_v2

# from akgr.tokenizer import QueryTokenizer, AnswersTokenizer, AnswersQueryTokenizer, ActionTokenizer

from datasets import Dataset
from akgr.utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr, list_to_str
from akgr.reproduction.config import ExperimentConfig
from akgr.reproduction.seed import derive_seed, make_generator

import pandas as pd

def pre_pre_processing(
        data: dict, pattern_str_2_id: dict,
        is_act:bool=False):
    # print(data_dict[split])
    df = pd.DataFrame.from_records(data)
    # print("#")
    # print(df)
    source = df['answers'].apply(ans_shift_indices)
    source = source.apply(list_to_str)
    # print(source)
    # By defualy, the target sequence is a query string
    target = df['query'].apply(qry_shift_indices)
    target = target.apply(list_to_str)
    if is_act: # action str
        target = target.apply(qry_str_2_actionstr)
    # print(target)
    pattern_id = df['pattern_str'].apply(lambda x: pattern_str_2_id[x])
    # print(pattern_id)
    # df = pd.DataFrame([source, target, pattern_id])
    # print("#")
    processed = pd.concat({
        'source': source,
        'target': target,
        'pattern_id': pattern_id}, axis=1)
    if 'record_id' in df:
        processed['record_id'] = df['record_id']
    return processed


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest_artifact(manifest_path, artifact):
    path = os.path.join(os.path.dirname(manifest_path), artifact['path'])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sampling artifact is missing: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != artifact['sha256']:
        raise ValueError(
            f"Sampling artifact hash mismatch for {path}: "
            f"manifest={artifact['sha256']} actual={actual_hash}"
        )
    records = []
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    if len(records) != artifact['count']:
        raise ValueError(
            f"Sampling artifact count mismatch for {path}: "
            f"manifest={artifact['count']} actual={len(records)}"
        )
    return records


def create_reproduction_dataset(
        experiment_config: ExperimentConfig,
        pattern_filtered,
        splits,
        is_act: bool,
        train_variant: str = 'base'):
    """Load hash-verified data selected by a strict reproduction manifest."""
    if train_variant not in {'base', 'merged'}:
        raise ValueError("train_variant must be base or merged")
    manifest_path = experiment_config.sampling_manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Sampling manifest not found: {manifest_path}. "
            "Run akgr.sampling.sample_parallel with the same experiment config first."
        )
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    deduplication = experiment_config.raw['sampling'].get('deduplication')
    expected = {
        'schema_version': 3 if deduplication is not None else 2,
        'dataset': experiment_config.dataset,
        'profile': experiment_config.experiment['profile'],
        'seed': experiment_config.seed,
        'data_hash': experiment_config.data_hash,
        'kg_hash': experiment_config.kg_hash,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Sampling manifest {key} mismatch: expected {value!r}, "
                f"found {manifest.get(key)!r}"
            )
    if deduplication is not None:
        manifest_deduplication = manifest.get('deduplication', {})
        if manifest_deduplication.get('policy') != deduplication:
            raise ValueError(
                'Sampling manifest deduplication mismatch: '
                f"expected {deduplication!r}, found {manifest_deduplication.get('policy')!r}"
            )
        if any(manifest_deduplication.get('post_dedup_overlap', {}).values()):
            raise ValueError('Sampling manifest reports cross-split supervision leakage')
    artifacts = manifest['artifacts']
    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered.index))
    dataset_dict = {}
    for split in splits:
        variant = train_variant if split == 'train' else 'base'
        if split not in artifacts.get(variant, {}):
            raise ValueError(f"Manifest has no {variant}.{split} artifact")
        raw_records = _load_manifest_artifact(
            str(manifest_path), artifacts[variant][split]
        )
        frame = pre_pre_processing(
            data=raw_records,
            pattern_str_2_id=pattern_str_2_id,
            is_act=is_act,
        )
        dataset_dict[split] = Dataset.from_pandas(frame, split=split, preserve_index=False)
    stats = manifest['stats']
    return dataset_dict, int(stats['nentity']), int(stats['nrelation'])
def new_create_dataset(dataname, scale, answer_size,
        pattern_filtered,
        data_root,
        splits,
        is_act:bool,
        # do_ordering:bool=False,
        # is_shared_ent:bool=False,
        # is_v2:bool=False
        ):

    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered.index))

    data_dict, nentity, nrelation = load_sampled_dataset(
        data_root=data_root,
        dataname=dataname,
        scale=scale,
        answer_size=answer_size,
        splits=splits,
        method='jsonl'
    )

    dataset_dict = {}
    for split in splits:
        df = pre_pre_processing(
            data=data_dict[split],
            pattern_str_2_id=pattern_str_2_id,
            is_act=is_act)
        # print(df)
        # exit()
        dataset_dict[split] = Dataset.from_pandas(df, split=split)

    return dataset_dict, nentity, nrelation
def new_create_dataloader(dataset_dict, batch_size:int, drop_last:bool=False,
                          shuffle:bool=True, seed:int=None, num_workers:int=4) :
    import warnings
    if drop_last:
        warnings.warn('drop_last is True')
    dataloader_dict = {}
    for split, dataset in dataset_dict.items():
        split_shuffle = shuffle and split == 'train'
        generator = None if seed is None else make_generator(derive_seed(seed, 'dataloader', split))
        dataloader_dict[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            generator=generator,
        )
    return dataloader_dict




if __name__ == '__main__':
    # config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    new_dataset_dict, nentity, nrelation = new_create_dataset('DBpedia50', 'debug', 32, is_act=True)
    print(new_dataset_dict['train'][:10])

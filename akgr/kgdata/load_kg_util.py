"""Knowledge-graph loading with reproducible splits and cache manifests.

The legacy project rebuilt a random split before it checked its pickle cache.
This module keeps the old ``load_kg(name)`` call working, while the reproduction
pipeline supplies a seed, data root, and semantic config hash and therefore gets
an isolated, verifiable cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from akgr.utils.nx_util import df_to_graph
from akgr.kgdata.kgclass import KG


TRIPLE_COLUMNS = ["head_id", "tail_id", "relation_id"]
CANONICAL_COLUMNS = ["head_id", "relation_id", "tail_id"]
KG_MANIFEST_SCHEMA = 1


def _module_distribution_version(module: Any) -> str:
    """Return a package version even when the module omits ``__version__``."""
    module_version = getattr(module, "__version__", None)
    if module_version is not None:
        return str(module_version)
    return importlib_metadata.version(module.__name__)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_triples(triples: pd.DataFrame | Iterable[Sequence[int]]) -> pd.DataFrame:
    """Return integer triples in a canonical, input-order-independent order."""
    if isinstance(triples, pd.DataFrame):
        missing = set(TRIPLE_COLUMNS) - set(triples.columns)
        if missing:
            raise ValueError(f"Triple frame is missing columns: {sorted(missing)}")
        frame = triples.loc[:, TRIPLE_COLUMNS].copy()
    else:
        frame = pd.DataFrame(list(triples), columns=TRIPLE_COLUMNS)
    for column in TRIPLE_COLUMNS:
        frame[column] = frame[column].astype("int64")
    return frame.sort_values(CANONICAL_COLUMNS, kind="mergesort").reset_index(drop=True)


def triples_hash(triples: pd.DataFrame) -> str:
    rows = canonicalize_triples(triples)[CANONICAL_COLUMNS].values.tolist()
    return _sha256_bytes(_canonical_json(rows))


def mapping_hash(mapping: Mapping[int, Any]) -> str:
    normalized = {str(int(key)): str(value) for key, value in mapping.items()}
    return _sha256_bytes(_canonical_json(normalized))


def split_triples(
    triples: pd.DataFrame | Iterable[Sequence[int]],
    seed: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, pd.DataFrame]:
    """Canonically shuffle and split triples into exclusive train/valid/test."""
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must contain three values summing to 1")
    frame = canonicalize_triples(triples)
    permutation = np.random.default_rng(int(seed)).permutation(len(frame))
    n_train = int(len(frame) * ratios[0])
    n_valid = int(len(frame) * ratios[1])
    boundaries = (n_train, n_train + n_valid)
    indices = {
        "train": permutation[: boundaries[0]],
        "valid": permutation[boundaries[0] : boundaries[1]],
        "test": permutation[boundaries[1] :],
    }
    return {
        split: canonicalize_triples(frame.iloc[split_indices])
        for split, split_indices in indices.items()
    }


def update_inverse_edges(
    rel_id2name: Mapping[int, Any], exclusive: Mapping[str, pd.DataFrame]
) -> tuple[dict[int, str], dict[int, int], dict[str, pd.DataFrame]]:
    """Add inverse edges after splitting, preserving split exclusivity."""
    new_id2name: dict[int, str] = {}
    rel_id2inv: dict[int, int] = {}
    for relation_id, name in sorted(rel_id2name.items()):
        forward, reverse = int(relation_id) * 2, int(relation_id) * 2 + 1
        new_id2name[forward] = f"+{name}"
        new_id2name[reverse] = f"-{name}"
        rel_id2inv[forward] = reverse
        rel_id2inv[reverse] = forward

    expanded: dict[str, pd.DataFrame] = {}
    for split, source in exclusive.items():
        forward = canonicalize_triples(source)
        forward["relation_id"] = forward["relation_id"] * 2
        reverse = source.rename(columns={"head_id": "tail_id", "tail_id": "head_id"})[
            TRIPLE_COLUMNS
        ].copy()
        reverse["relation_id"] = reverse["relation_id"] * 2 + 1
        expanded[split] = canonicalize_triples(pd.concat([forward, reverse], ignore_index=True))
    return new_id2name, rel_id2inv, expanded


def _cumulative_frames(exclusive: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "train": canonicalize_triples(exclusive["train"]),
        "valid": canonicalize_triples(pd.concat([exclusive["train"], exclusive["valid"]], ignore_index=True)),
        "test": canonicalize_triples(
            pd.concat([exclusive["train"], exclusive["valid"], exclusive["test"]], ignore_index=True)
        ),
    }


def build_kg_from_triples(
    triples: pd.DataFrame | Iterable[Sequence[int]],
    *,
    num_ent: int,
    rel_id2name: Mapping[int, Any],
    ent_id2name: Mapping[int, Any] | None = None,
    seed: int = 42,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    reverse_edges_flag: bool = True,
) -> tuple[KG, dict[str, Any]]:
    """Pure synthetic-friendly KG builder used by the loader and tests."""
    raw = canonicalize_triples(triples)
    exclusive = split_triples(raw, seed, split_ratios)
    original_relations = {int(key): str(value) for key, value in rel_id2name.items()}
    if reverse_edges_flag:
        final_relations, rel_id2inv, graph_exclusive = update_inverse_edges(original_relations, exclusive)
    else:
        final_relations = original_relations
        rel_id2inv = {}
        graph_exclusive = {split: canonicalize_triples(frame) for split, frame in exclusive.items()}
    cumulative = _cumulative_frames(graph_exclusive)
    graphs = {split: df_to_graph(frame[TRIPLE_COLUMNS]) for split, frame in cumulative.items()}
    kg = KG(
        num_ent=int(num_ent),
        num_rel=len(final_relations),
        ent_id2name=dict(ent_id2name or {index: str(index) for index in range(int(num_ent))}),
        rel_id2name=final_relations,
        rel_id2inv=rel_id2inv,
        graphs=graphs,
    )
    metadata = {
        "raw": {"count": len(raw), "sha256": triples_hash(raw)},
        "exclusive_splits": {
            split: {"count": len(frame), "sha256": triples_hash(frame)}
            for split, frame in exclusive.items()
        },
        "cumulative_graphs": {
            split: {"count": len(frame), "sha256": triples_hash(frame)}
            for split, frame in cumulative.items()
        },
        "entity_mapping_sha256": mapping_hash(kg.ent_id2name),
        "relation_mapping_sha256": mapping_hash(kg.rel_id2name),
    }
    return kg, metadata


def _dataset_class(dataname: str):
    import pykeen.datasets as datasets

    supported = {
        "YAGO310": datasets.YAGO310,
        "FB15k-237": datasets.FB15k237,
        "DBpedia50": datasets.DBpedia50,
        "BioKG": datasets.BioKG,
        "PharmKG8k": datasets.PharmKG8k,
        "WN18RR": datasets.WN18RR,
        "OGBWikiKG2": datasets.OGBWikiKG2,
    }
    try:
        return supported[dataname]
    except KeyError as exc:
        raise ValueError(f"Dataset {dataname!r} is not supported") from exc


def _load_raw_dataset(dataname: str) -> tuple[Any, str, pd.DataFrame, dict[int, Any], dict[int, Any]]:
    import pykeen
    import pykeen.utils as pk_utils

    dataset = _dataset_class(dataname)(create_inverse_triples=False)
    frames = []
    for split in ("training", "validation", "testing"):
        factory = dataset.factory_dict[split]
        frame = factory.tensor_to_df(factory.mapped_triples)[TRIPLE_COLUMNS]
        frames.append(frame)
    raw = canonicalize_triples(pd.concat(frames, ignore_index=True))
    entities = pk_utils.invert_mapping(dataset.entity_to_id)
    relations = pk_utils.invert_mapping(dataset.relation_to_id)
    return pykeen, type(dataset).__name__, raw, entities, relations


def _cache_paths(
    data_root: Path,
    dataname: str,
    seed: int | None,
    reverse_edges_flag: bool,
    split_ratios: Sequence[float],
    semantic_hash: str | None,
) -> tuple[Path, Path]:
    directory = data_root / dataname
    directory.mkdir(parents=True, exist_ok=True)
    if seed is None and semantic_hash is None:
        stem = dataname
    else:
        request = {
            "dataset": dataname,
            "seed": int(seed if seed is not None else 0),
            "reverse_edges": bool(reverse_edges_flag),
            "split_ratios": [float(value) for value in split_ratios],
            "semantic_hash": semantic_hash,
        }
        stem = f"{dataname}-kg-{_sha256_bytes(_canonical_json(request))[:16]}"
    return directory / f"{stem}.pkl", directory / f"{stem}.manifest.json"


def _request_matches(
    manifest: Mapping[str, Any],
    *,
    dataname: str,
    seed: int | None,
    reverse_edges_flag: bool,
    split_ratios: Sequence[float],
    semantic_hash: str | None,
) -> bool:
    return (
        manifest.get("schema_version") == KG_MANIFEST_SCHEMA
        and manifest.get("dataset") == dataname
        and manifest.get("seed") == (int(seed) if seed is not None else None)
        and manifest.get("reverse_edges") is bool(reverse_edges_flag)
        and manifest.get("split_ratios") == [float(value) for value in split_ratios]
        and manifest.get("semantic_hash") == semantic_hash
    )


def _read_cache(cache_path: Path, manifest_path: Path, request: Mapping[str, Any]) -> KG | None:
    cache_exists = cache_path.exists()
    manifest_exists = manifest_path.exists()
    if not cache_exists and not manifest_exists:
        return None
    if cache_exists != manifest_exists:
        raise ValueError(
            f"Incomplete KG cache pair; cache={cache_exists}, manifest={manifest_exists}: "
            f"{cache_path}, {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _request_matches(manifest, **request):
        raise ValueError(f"KG cache manifest does not match the requested configuration: {manifest_path}")
    if manifest.get("cache", {}).get("sha256") != sha256_file(cache_path):
        raise ValueError(f"KG cache checksum mismatch: {cache_path}")
    with cache_path.open("rb") as handle:
        kg = pickle.load(handle)
    kg.cache_path = cache_path
    kg.cache_manifest_path = manifest_path
    kg.cache_manifest = manifest
    return kg


def load_kg(
    dataname: str,
    reverse_edges_flag: bool = True,
    id_map_only: bool = False,
    *,
    data_root: str | Path = "./sampled_data",
    seed: int | None = None,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    semantic_hash: str | None = None,
    offline: bool = False,
) -> KG | dict[str, dict[int, Any]]:
    """Load a KG cache first, or build one from PyKEEN with a fixed split.

    ``offline=True`` never instantiates a PyKEEN dataset. It therefore succeeds
    only when the exact requested cache and manifest already exist.
    """
    root = Path(data_root).expanduser().resolve()
    cache_path, manifest_path = _cache_paths(
        root, dataname, seed, reverse_edges_flag, split_ratios, semantic_hash
    )
    request = {
        "dataname": dataname,
        "seed": seed,
        "reverse_edges_flag": reverse_edges_flag,
        "split_ratios": split_ratios,
        "semantic_hash": semantic_hash,
    }
    if not id_map_only:
        cached = _read_cache(cache_path, manifest_path, request)
        if cached is not None:
            print(f"# KG loaded from {cache_path}")
            return cached
    if offline:
        raise FileNotFoundError(f"No valid offline KG cache for request: {manifest_path}")

    pykeen, dataset_class_name, raw, entities, relations = _load_raw_dataset(dataname)
    if id_map_only:
        return {"ent_id2name": entities, "rel_id2name": relations}
    effective_seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy)
    kg, metadata = build_kg_from_triples(
        raw,
        num_ent=len(entities),
        rel_id2name=relations,
        ent_id2name=entities,
        seed=effective_seed,
        split_ratios=split_ratios,
        reverse_edges_flag=reverse_edges_flag,
    )
    with cache_path.open("wb") as handle:
        pickle.dump(kg, handle)
    pykeen_version = _module_distribution_version(pykeen)
    manifest = {
        "schema_version": KG_MANIFEST_SCHEMA,
        "dataset": dataname,
        "dataset_class": dataset_class_name,
        "pykeen_version": pykeen_version,
        "source": {
            "provider": "pykeen",
            "dataset": dataname,
            "dataset_class": dataset_class_name,
            "pykeen_version": pykeen_version,
        },
        "seed": int(seed) if seed is not None else None,
        "effective_seed": effective_seed,
        "split_ratios": [float(value) for value in split_ratios],
        "reverse_edges": bool(reverse_edges_flag),
        "split": {
            "algorithm": "canonical-sort+numpy-default-rng-permutation-v1",
            "seed": int(seed) if seed is not None else None,
            "effective_seed": effective_seed,
            "ratios": [float(value) for value in split_ratios],
        },
        "inverse_edges": {
            "enabled": bool(reverse_edges_flag),
            "applied_after_exclusive_split": True,
        },
        "semantic_hash": semantic_hash,
        "stats": {"nentity": kg.num_ent, "nrelation": kg.num_rel},
        **metadata,
        "mappings": {
            "entity": {
                "count": len(kg.ent_id2name),
                "sha256": metadata["entity_mapping_sha256"],
            },
            "relation": {
                "count": len(kg.rel_id2name),
                "sha256": metadata["relation_mapping_sha256"],
            },
        },
        "cache": {"path": cache_path.name, "sha256": sha256_file(cache_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    kg.cache_path = cache_path
    kg.cache_manifest_path = manifest_path
    kg.cache_manifest = manifest
    print(f"# KG saved to {cache_path}")
    return kg


def load_fb15k237_ent_2idname(ent_id2name: Mapping[int, str]) -> dict[int, str]:
    """Legacy optional label helper; sampling no longer calls it eagerly."""
    path = Path("akgr/metadata/FB15k_mid2name.txt")
    with path.open("r", encoding="utf-8") as handle:
        mid2name = dict(csv.reader(handle, delimiter="\t"))
    return {index: mid2name[name] for index, name in ent_id2name.items()}


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataname", default="YAGO310")
    parser.add_argument("--data-root", default="./sampled_data")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = my_parse_args()
    load_kg(args.dataname, data_root=args.data_root, seed=args.seed, offline=args.offline)

"""Perceptual image similarity and group construction utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image


def compute_perceptual_hashes(
    image_paths: Iterable[Path],
    hash_size: int = 8,
) -> dict[str, imagehash.ImageHash]:
    """Compute pHash values keyed by image stem."""
    if hash_size < 2:
        raise ValueError("hash_size must be at least 2")

    hashes: dict[str, imagehash.ImageHash] = {}
    for path in image_paths:
        with Image.open(path) as image:
            hashes[path.stem] = imagehash.phash(image.convert("RGB"), hash_size=hash_size)
    return hashes


def find_similar_pairs(
    hashes: Mapping[str, imagehash.ImageHash],
    max_distance: int = 6,
) -> pd.DataFrame:
    """Return all pHash pairs within a Hamming-distance threshold."""
    if max_distance < 0:
        raise ValueError("max_distance cannot be negative")

    items = sorted(hashes.items())
    pairs: list[dict[str, int | str]] = []
    for left_index, (left_id, left_hash) in enumerate(items):
        for right_id, right_hash in items[left_index + 1 :]:
            distance = left_hash - right_hash
            if distance <= max_distance:
                pairs.append(
                    {
                        "left_id": left_id,
                        "right_id": right_id,
                        "phash_distance": distance,
                    }
                )
    return pd.DataFrame(
        pairs,
        columns=["left_id", "right_id", "phash_distance"],
    ).sort_values("phash_distance", ignore_index=True)


def build_similarity_groups(
    image_ids: Iterable[str],
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build connected components from reviewed similar-image pairs."""
    ids = sorted(set(image_ids))
    parent = {image_id: image_id for image_id in ids}

    def find(image_id: str) -> str:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(left_id: str, right_id: str) -> None:
        left_root, right_root = find(left_id), find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in pairs.itertuples(index=False):
        if row.left_id not in parent or row.right_id not in parent:
            raise ValueError("Every pair must reference a known image ID")
        union(row.left_id, row.right_id)

    roots = {image_id: find(image_id) for image_id in ids}
    root_counts = Counter(roots.values())
    grouped_roots = sorted(root for root, count in root_counts.items() if count > 1)
    group_ids = {root: index for index, root in enumerate(grouped_roots)}
    return pd.DataFrame(
        {
            "Image_ID": ids,
            "similarity_group": [group_ids.get(roots[image_id], -1) for image_id in ids],
        }
    )


def select_pair_rows(pairs: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    """Select manually reviewed candidate pairs by row index."""
    return pairs.loc[list(indices)].reset_index(drop=True)


def select_reviewed_pairs(
    pairs: pd.DataFrame,
    reviewed_pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Select reviewed pairs by image IDs, independent of row ordering."""
    canonical = {tuple(sorted(pair)) for pair in reviewed_pairs}
    available = {
        tuple(sorted((row.left_id, row.right_id))): index
        for index, row in enumerate(pairs.itertuples(index=False))
    }
    missing = canonical - set(available)
    if missing:
        raise ValueError(f"Reviewed pairs are absent from candidates: {sorted(missing)}")
    indices = [available[pair] for pair in sorted(canonical)]
    return pairs.iloc[indices].sort_values("phash_distance", ignore_index=True)

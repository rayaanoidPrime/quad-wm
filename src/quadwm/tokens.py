"""Frozen-encoder visual tokens: image preparation and the on-disk token cache.

Shared by training and evaluation. Only models with ``frozen_visual_encoder``
(the V-JEPA baseline) use tokens; end-to-end models take raw images.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def prepare_images(images: torch.Tensor, device: torch.device, size: int) -> torch.Tensor:
    batch, frames, channels, height, width = images.shape
    # This ROCm stack hands back an all-NaN tensor for the H2D copy of a large
    # pinned float32 tensor (HIP pinned-allocator bug), synchronously or not.
    # An unpinned clone is safe, so drop the pin before copying. Eval loaders
    # pin their batches; see collect_latents.
    if images.is_pinned():
        images = images.clone()
    images = images.to(device).div(255.0)
    images = images.reshape(batch * frames, channels, height, width)
    images = torch.nn.functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((images - mean) / std).reshape(batch, frames, channels, size, size)


class TokenCache:
    def __init__(self, root: Path, mission: str):
        self.root = root
        self.mission = mission
        metadata = json.loads((root / f"{mission}.json").read_text(encoding="utf-8"))
        self.ids = np.asarray(metadata["image_ids"], dtype=np.int64)
        self.lookup = {int(image_id): index for index, image_id in enumerate(self.ids)}
        self.tokens = np.memmap(
            root / f"{mission}.mmap",
            mode="r",
            dtype=np.float16,
            shape=tuple(metadata["shape"]),
        )

    def get(self, image_ids: np.ndarray) -> np.ndarray:
        positions = [self.lookup[int(image_id)] for image_id in image_ids.reshape(-1)]
        return np.asarray(self.tokens[positions]).reshape(
            *image_ids.shape, self.tokens.shape[1], self.tokens.shape[2]
        )


def _cache_valid(root: Path, mission: str, image_ids: list[int], shape: tuple[int, ...]) -> bool:
    """True when an existing cache has the same token dims and covers every id.

    Superset reuse matters when an eval dataset indexes a subset of a mission
    that training already cached -- rebuilding would clobber the train cache.
    """
    metadata_path = root / f"{mission}.json"
    mmap_path = root / f"{mission}.mmap"
    if not metadata_path.is_file() or not mmap_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    existing_shape = tuple(metadata["shape"])
    return existing_shape[1:] == shape[1:] and set(image_ids) <= set(metadata["image_ids"])


def _mission_image_ids(dataset, reader_index: int) -> list[int]:
    """Sorted unique image ids the reader contributes to the dataset's sequences."""
    return sorted(
        {
            image_id
            for index in dataset.index
            if index[0] == reader_index
            for image_id in index[1]
        }
    )


@torch.no_grad()
def build_token_cache(
    dataset,
    encoder,
    cache_root: Path,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    encoder.eval().to(device)
    for reader_index, reader in enumerate(dataset.readers):
        mission = reader.mission_dir.name
        image_ids = _mission_image_ids(dataset, reader_index)
        if not image_ids:
            # A short or fully-invalid mission contributes no sequences, so there
            # is nothing to cache (its reader is simply not used by the loader).
            continue
        first = prepare_images(
            torch.from_numpy(
                np.stack([reader.load_image(i) for i in image_ids[:1]])
            ).permute(0, 3, 1, 2).unsqueeze(1),
            device,
            image_size,
        )[:, 0]
        first_tokens = encoder(first).float().cpu().numpy().astype(np.float16)
        shape = (len(image_ids), *first_tokens.shape[1:])
        if _cache_valid(cache_root, mission, image_ids, shape):
            continue
        metadata_path = cache_root / f"{mission}.json"
        if metadata_path.is_file():
            # Rebuild as the union with what is cached, so a dataset with other
            # windows (protocol eval vs training) never drops the other's ids.
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if tuple(existing["shape"][1:]) == shape[1:]:
                image_ids = sorted(set(image_ids) | set(existing["image_ids"]))
                shape = (len(image_ids), *shape[1:])
        print(
            f"stage=feature_cache mission={mission} images={len(image_ids)} "
            f"shape={tuple(shape)}",
            flush=True,
        )
        temporary = cache_root / f"{mission}.mmap.part"
        metadata_temporary = cache_root / f"{mission}.json.part"
        required_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
        free = shutil.disk_usage(cache_root).free
        if required_bytes > free * 0.9:
            raise RuntimeError(
                f"not enough disk for token cache of {mission}: "
                f"need {required_bytes / 1024**3:.1f} GiB, free {free / 1024**3:.1f} GiB. "
                "Set cache.mode=off or point QUADWM_CACHE_ROOT at a larger filesystem."
            )
        try:
            tokens = np.memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
            for start in range(0, len(image_ids), batch_size):
                ids = image_ids[start : start + batch_size]
                images = np.stack([reader.load_image(image_id) for image_id in ids])
                images = prepare_images(
                    torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(1),
                    device,
                    image_size,
                )[:, 0]
                tokens[start : start + len(ids)] = encoder(images).float().cpu().numpy().astype(np.float16)
            tokens.flush()
            del tokens  # close the mapping; Windows cannot rename an open file
            metadata_temporary.write_text(
                json.dumps({"image_ids": image_ids, "shape": list(shape)}) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, cache_root / f"{mission}.mmap")
            os.replace(metadata_temporary, cache_root / f"{mission}.json")
        finally:
            # Never leave a multi-GiB .part behind if the job dies mid-write.
            temporary.unlink(missing_ok=True)
            metadata_temporary.unlink(missing_ok=True)
        print(f"stage=feature_cache mission={mission} status=done", flush=True)


def cache_fits(datasets, cache_root: Path, tokens: int, dim: int) -> bool:
    ids = sum(
        len(_mission_image_ids(dataset, reader_index))
        for dataset in datasets
        for reader_index in range(len(dataset.readers))
    )
    required = ids * tokens * dim * 2
    free = shutil.disk_usage(cache_root.parent).free
    print(f"feature cache estimate: {required / 1024**3:.1f} GiB; disk free: {free / 1024**3:.1f} GiB")
    return required < free * 0.8


def cached_tokens(batch: dict, caches: list[TokenCache | None], device: torch.device) -> torch.Tensor:
    mission_ids = batch["mission_idx"].tolist()
    image_ids = batch["image_ids"].numpy()
    rows = []
    for row, mission_id in enumerate(mission_ids):
        cache = caches[mission_id]
        if cache is None:
            raise RuntimeError(f"no token cache for mission index {mission_id}")
        rows.append(cache.get(image_ids[row]))
    return torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)


def load_token_caches(cache_root: Path, readers) -> list[TokenCache | None]:
    """Per-mission caches, aligned with reader order; None where nothing was cached."""
    return [
        TokenCache(cache_root, reader.mission_dir.name)
        if (cache_root / f"{reader.mission_dir.name}.json").is_file()
        else None
        for reader in readers
    ]


def frozen_tokens(model, batch: dict, device: torch.device, caches: list[TokenCache | None] | None = None
                  ) -> Tensor | None:
    """Frozen-encoder tokens [B, T, N, D] for ``batch`` (read from ``caches`` when given); None for
    end-to-end models."""
    module = getattr(model, "module", model)  # unwrap DDP
    if not module.frozen_visual_encoder:
        return None
    if caches is not None:
        return cached_tokens(batch, caches, device)
    images = prepare_images(batch["images"], device, module.image_size)
    return module.encode_visual(images.flatten(0, 1)).unflatten(0, images.shape[:2])


def model_inputs(model, batch: dict, device: torch.device, caches: list[TokenCache | None] | None = None
                 ) -> tuple[dict, Tensor | None]:
    """(batch on ``device``, frozen tokens or None): the arguments of ``model.encode_frames``."""
    tokens = frozen_tokens(model, batch, device, caches)
    keys = ("proprio", "actions") if tokens is not None else ("proprio", "actions", "images")
    return {key: batch[key].to(device, non_blocking=True) for key in keys if key in batch}, tokens

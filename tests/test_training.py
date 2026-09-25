import json

import numpy as np
import pytest
import torch

from quadwm.training import _build_token_cache


class _RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(images.shape))
        return torch.ones(images.shape[0], 4, 8)


class _MissionDir:
    name = "mission-a"


class _Reader:
    mission_dir = _MissionDir()

    def load_image(self, _: int) -> np.ndarray:
        # GrandTour images load as HWC uint8, as imageio returns them.
        return np.zeros((1080, 1440, 3), dtype=np.uint8)


class _Dataset:
    readers = [_Reader()]
    index = [(0, (0, 1, 2))]


def test_token_cache_feeds_chw_images_to_encoder(tmp_path):
    encoder = _RecordingEncoder()

    _build_token_cache(_Dataset(), encoder, tmp_path, torch.device("cpu"), batch_size=2, image_size=64)

    # Regression: the cache path previously passed HWC tensors straight to
    # _prepare_images, which expects NCHW and would interpolate across the
    # colour channels.
    assert encoder.input_shapes
    assert all(shape[1] == 3 for shape in encoder.input_shapes)
    assert all(shape[-2:] == (64, 64) for shape in encoder.input_shapes)

    metadata = (tmp_path / "mission-a.json").read_text(encoding="utf-8")
    assert '"shape": [3, 4, 8]' in metadata


def test_cache_valid_accepts_superset_of_ids(tmp_path):
    from quadwm.training import _cache_valid

    (tmp_path / "mission-a.mmap").write_bytes(b"")
    (tmp_path / "mission-a.json").write_text(
        json.dumps({"image_ids": [1, 2, 3], "shape": [3, 4, 8]}), encoding="utf-8"
    )

    # A subset of the cached ids reuses the cache instead of clobbering it.
    assert _cache_valid(tmp_path, "mission-a", [2, 3], (2, 4, 8))
    # Missing id or mismatched token dims must not reuse.
    assert not _cache_valid(tmp_path, "mission-a", [2, 9], (2, 4, 8))
    assert not _cache_valid(tmp_path, "mission-a", [2, 3], (2, 5, 8))


def test_token_cache_removes_partial_files_on_failure(tmp_path):
    class _FailingEncoder(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("encoder blew up")

    with pytest.raises(RuntimeError, match="encoder blew up"):
        _build_token_cache(
            _Dataset(), _FailingEncoder(), tmp_path, torch.device("cpu"), batch_size=2, image_size=64
        )

    assert list(tmp_path.glob("*.part")) == []
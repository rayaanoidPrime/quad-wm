import numpy as np
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
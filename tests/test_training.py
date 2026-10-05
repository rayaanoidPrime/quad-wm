import json
from pathlib import Path

import numpy as np
import pytest
import torch

from quadwm.tokens import build_token_cache


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

    build_token_cache(_Dataset(), encoder, tmp_path, torch.device("cpu"), batch_size=2, image_size=64)

    # Regression: the cache path previously passed HWC tensors straight to
    # _prepare_images, which expects NCHW and would interpolate across the
    # colour channels.
    assert encoder.input_shapes
    assert all(shape[1] == 3 for shape in encoder.input_shapes)
    assert all(shape[-2:] == (64, 64) for shape in encoder.input_shapes)

    metadata = (tmp_path / "mission-a.json").read_text(encoding="utf-8")
    assert '"shape": [3, 4, 8]' in metadata


def test_cache_valid_accepts_superset_of_ids(tmp_path):
    from quadwm.tokens import _cache_valid

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
        build_token_cache(
            _Dataset(), _FailingEncoder(), tmp_path, torch.device("cpu"), batch_size=2, image_size=64
        )

    assert list(tmp_path.glob("*.part")) == []

def test_token_cache_rebuild_keeps_previously_cached_ids(tmp_path):
    """The protocol eval caches other windows than training; neither may drop the other's ids."""
    build_token_cache(_Dataset(), _RecordingEncoder(), tmp_path, torch.device("cpu"), batch_size=2, image_size=64)

    other_windows = _Dataset()
    other_windows.index = [(0, (3, 4))]
    build_token_cache(other_windows, _RecordingEncoder(), tmp_path, torch.device("cpu"), batch_size=2,
                       image_size=64)

    metadata = json.loads((tmp_path / "mission-a.json").read_text(encoding="utf-8"))
    assert metadata["image_ids"] == [0, 1, 2, 3, 4]
    assert metadata["shape"] == [5, 4, 8]


def test_run_log_writes_the_metrics_jsonl_schema_eval_reads(tmp_path):
    """protocol.training_compute reads step timestamps from metrics.jsonl (EV7 GPU-hours)."""
    from quadwm.training import _RunLog

    class _Wandb:
        def __init__(self):
            self.logged = []

        def log(self, values, step):
            self.logged.append((values, step))

        def finish(self):
            pass

    wandb_run = _Wandb()
    log = _RunLog(tmp_path, wandb_run, enabled=True)
    log.record({"loss": 1.0, "epoch": 0}, 1, to_wandb=False)
    log.record({"eval/visual_mse": 2.0}, 1)
    log.close()

    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [set(line) for line in lines] == [{"loss", "epoch", "step", "timestamp"},
                                             {"eval/visual_mse", "step", "timestamp"}]
    assert wandb_run.logged == [({"eval/visual_mse": 2.0}, 1)]
    assert not (tmp_path / "other").exists()
    _RunLog(tmp_path / "other", None, enabled=False).record({"loss": 1.0}, 1)  # non-zero ranks write nothing


@pytest.mark.filterwarnings("ignore::UserWarning")  # CUDA autocast is disabled on CPU-only hosts
def test_held_out_eval_drives_both_models_through_one_interface():
    from quadwm.models import JEPAWorldModel, LeWorldModel
    from quadwm.training import _evaluate

    lewm = LeWorldModel(image_size=16, patch_size=8, encoder_dim=16, encoder_depth=1, encoder_heads=2,
                        latent_dim=8, predictor_dim=16, predictor_depth=1, predictor_heads=2,
                        predictor_window=2, proprio_dim=33, action_dim=6, context_steps=2,
                        rollout_steps=1, sigreg_slices=8)
    jepa = JEPAWorldModel(visual_dim=8, proprio_dim=33, proprio_embed_dim=4, action_dim=6, tokens_per_frame=4,
                          predictor_depth=1, predictor_heads=2, context_steps=2, rollout_context=2)

    class _Cache:
        def get(self, image_ids):
            return np.ones((*image_ids.shape, 4, 8), dtype=np.float16)

    batch = {"proprio": torch.randn(2, 3, 33), "actions": torch.randn(2, 2, 6), "images": torch.rand(2, 3, 2, 16, 16),
             "mission_idx": torch.zeros(2, dtype=torch.long), "image_ids": torch.arange(6).view(2, 3)}
    cpu = torch.device("cpu")
    lewm_metrics = _evaluate(lewm.train(), [dict(batch)], None, cpu, torch.bfloat16, max_batches=1)
    jepa_metrics = _evaluate(jepa, [dict(batch)], [_Cache()], cpu, torch.bfloat16, max_batches=1)

    assert "step_1/persistence_mse" in lewm_metrics and lewm.training  # training mode restored
    assert "step_1/persistence_visual_mse" in jepa_metrics
    # forward is the training loss for both, with the same (batch, visual_tokens) signature
    tokens = torch.ones(2, 3, 4, 8)
    assert torch.equal(jepa(batch, tokens)["loss"], jepa.loss(tokens, batch["proprio"], batch["actions"])["loss"])


def test_resume_last_follows_the_run_name(tmp_path):
    """``resume: last`` cannot point at another run's checkpoint after ``name`` is bumped."""
    from quadwm.training import _resume_path

    config = {"run_root": str(tmp_path), "name": "jepa-baseline-v3", "training": {"resume": "last"}}
    assert _resume_path(config) == tmp_path / "jepa-baseline-v3" / "last.pt"
    config["training"]["resume"] = "elsewhere/last.pt"
    assert _resume_path(config) == Path("elsewhere/last.pt")
    config["training"]["resume"] = ""
    assert _resume_path(config) is None

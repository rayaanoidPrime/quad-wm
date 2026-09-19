import json
import sys
import types
from pathlib import Path

from quadwm.cli import smoke
from quadwm.config import load_config


def test_smoke_writes_metadata_and_metrics(tmp_path):
    output_dir = tmp_path / "run"
    jepa_smoke_config = load_config(path=Path("configs/jepa-wm/smoke.yaml"))
    jepa_smoke_config["run_root"]  = output_dir
    jepa_smoke_config["steps"] = 3
    smoke(jepa_smoke_config)

    metadata = json.loads((output_dir /"smoke"/ "run_metadata.json").read_text())
    metrics = [json.loads(line) for line in (output_dir / "smoke"/ "metrics.jsonl").read_text().splitlines()]

    assert metadata["slurm_job_id"] == "local"
    assert [item["step"] for item in metrics] == [1, 2, 3]
    assert metrics[0]["loss"] > metrics[-1]["loss"]


def test_smoke_resolves_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("QUADWM_DATA_ROOT", "/scratch/example/data")
    config = load_config(path=Path("configs/jepa-wm/smoke.yaml"))
    assert config["data_root"] == "/scratch/example/data"


def test_wandb_init_uses_config_without_network(monkeypatch, tmp_path):
    calls = {}

    class FakeRun:
        id = "run-123"
        url = "https://wandb.example/run-123"

    def fake_init(**kwargs):
        calls.update(kwargs)
        return FakeRun()

    fake_wandb = types.SimpleNamespace(
        init=fake_init,
        Settings=lambda **kwargs: {"settings": kwargs},
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    from quadwm.utils.wandb import init_wandb

    run = init_wandb(
        {
            "config_path": "configs/jepa-wm/smoke.yaml",
            "wandb": {
                "enabled": True,
                "project": "quad-wm-test",
                "name": "smoke-test",
                "mode": "offline",
            },
        },
        metadata={"slurm_job_id": "123"},
        run_dir=tmp_path,
    )

    assert run.id == "run-123"
    assert calls["project"] == "quad-wm-test"
    assert calls["mode"] == "offline"
    assert calls["dir"] == str(tmp_path)
    assert calls["settings"]["settings"]["start_method"] == "thread"
    assert calls["config"]["runtime"]["slurm_job_id"] == "123"

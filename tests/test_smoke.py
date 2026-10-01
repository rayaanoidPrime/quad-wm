import sys
import types
from pathlib import Path

from quadwm.config import load_config


def test_config_resolves_environment_defaults(monkeypatch):
    monkeypatch.setenv("GRANDTOUR_ROOT", "/scratch/example/data")
    config = load_config(path=Path("configs/data/grandtour_smoke.yaml"))
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
            "config_path": "configs/jepa-wm/baseline_smoke.yaml",
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

def test_config_expands_home_in_environment_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("QUADWM_TEST_ROOT", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_root: ${QUADWM_TEST_ROOT:-~/quad-wm-storage/runs}\n", encoding="utf-8")

    run_root = load_config(config_path)["run_root"]

    assert not run_root.startswith("~")
    assert Path(run_root) == Path.home() / "quad-wm-storage" / "runs"

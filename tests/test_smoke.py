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


def test_config_expands_home_in_environment_values(monkeypatch, tmp_path):
    # A literal "~" can arrive through the environment (e.g. an sbatch --export);
    # it must not create a ./~ directory inside the checkout.
    monkeypatch.setenv("QUADWM_TEST_ROOT", "~/quad-wm-storage/runs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_root: ${QUADWM_TEST_ROOT}\n", encoding="utf-8")

    run_root = load_config(config_path)["run_root"]

    assert not run_root.startswith("~")
    assert Path(run_root) == Path.home() / "quad-wm-storage" / "runs"


def test_eval_logs_the_protocol_to_wandb(monkeypatch, tmp_path):
    from quadwm.evaluation import protocol

    calls = {}

    class FakeRun:
        id = "eval-run-1"
        url = "https://wandb.example/eval-run-1"

        def log(self, metrics):
            calls["metrics"] = metrics

        def finish(self):
            calls["finished"] = True

    def fake_init(cfg, *, metadata=None, run_dir=None):
        calls["cfg"] = cfg
        return FakeRun()

    monkeypatch.setattr(protocol, "init_wandb", fake_init)
    result = {
        "data": {"windows": {"probe": 4, "eval": 4}},
        "gait_cycle": {"seconds": 0.8, "ticks": 4},
        "probes": {"linear": {
            "quality": {"all": {"r2": 0.5, "pearson": 0.7}},
            "eps_k": {"1": {"model": {"all": 1.0}, "persistence": {"all": 2.0},
                             "encoded_floor": {"all": 0.5}}},
        }},
        "compute": {"rollout_throughput_fps": 10.0, "single_step_latency_ms": 3.0,
                    "parameters_M": {"world_model_trainable": 1.0}},
        "model": {"epoch": 2, "latent_dim": 8},
        "metadata": {},
    }
    config = {"name": "run-x", "wandb": {"enabled": True, "project": "p", "group": "g", "tags": ["t"]}}

    protocol._log_eval_wandb(config, {"name": "protocol-v1"}, result, tmp_path)

    assert calls["cfg"]["wandb"]["job_type"] == "eval"
    assert calls["cfg"]["wandb"]["group"] == "g"
    assert "eval-protocol-v1" in calls["cfg"]["wandb"]["tags"]
    assert calls["metrics"]["probe/linear/r2/all"] == 0.5
    assert calls["metrics"]["eps_k/linear/model/1"] == 1.0
    assert result["metadata"]["wandb_url"] == "https://wandb.example/eval-run-1"
    assert calls["finished"] is True


def test_eval_num_workers_is_zero_only_for_full_res_rgb():
    from quadwm.evaluation.protocol import _eval_num_workers

    eval_config = {"num_workers": 8}
    assert _eval_num_workers(eval_config, {"observation": "depth_plus_proprioception"}) == 8
    assert _eval_num_workers(eval_config, {"observation": "rgb_plus_proprioception"}) == 0

import json
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

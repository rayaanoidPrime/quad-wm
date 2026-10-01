import json

import numpy as np
import pytest
import torch

from quadwm.data.grandtour import STATE_LAYOUT, state_vectors
from quadwm.evaluation import (
    build_report,
    comparability_problems,
    fit_probe,
    gait_cycle_seconds,
    holm,
    mann_whitney,
    protocol_latents,
    r2_and_pearson,
    rollout_errors,
    select_anchor_windows,
)
from quadwm.models import JEPAWorldModel, LeWorldModel


def test_rollout_errors_normalize_per_component_and_split_by_layout():
    target = torch.zeros(2, 40)
    predicted = torch.zeros(2, 40)
    predicted[:, STATE_LAYOUT["base_pos"]] = 3.0  # 3 dims x 3 / sigma 3 -> norm sqrt(3)
    predicted[:, STATE_LAYOUT["joint_pos"].start] = 2.0  # 1 dim x 2 / sigma 1 -> 2
    sigma = torch.ones(40)
    sigma[STATE_LAYOUT["base_pos"]] = 3.0

    errors = rollout_errors(predicted, target, sigma)

    assert torch.allclose(errors["base_pos"], torch.full((2,), 3**0.5))
    assert torch.allclose(errors["joint_pos"], torch.full((2,), 2.0))
    assert torch.allclose(errors["all"], torch.full((2,), (3 + 4) ** 0.5))
    assert torch.allclose(errors["all_excl_base_pos"], torch.full((2,), 2.0))
    assert torch.allclose(errors["contacts"], torch.zeros(2))


def test_r2_and_pearson_perfect_fit_and_skips_constant_dims():
    target = torch.randn(100, 40, generator=torch.Generator().manual_seed(0))
    target[:, STATE_LAYOUT["contacts"]] = 1.0  # never varies on this eval set

    quality = r2_and_pearson(target.clone(), target)

    assert quality["joint_pos"]["r2"] == pytest.approx(1.0)
    assert quality["joint_pos"]["pearson"] == pytest.approx(1.0)
    assert np.isnan(quality["contacts"]["r2"])
    assert quality["all"]["r2"] == pytest.approx(1.0)


def test_select_anchor_windows_matches_every_observation_and_thins():
    own = np.arange(0.0, 60.0, 0.2)
    other = own[(own < 20.0) | (own > 30.0)] + 0.05  # the other camera has a 10 s hole

    chosen = select_anchor_windows(own, [other], stride_s=5.0, tolerance_s=0.1, max_windows=0)
    times = own[chosen]

    assert len(times) > 5
    assert (np.minimum(times % 5.0, 5.0 - times % 5.0) <= 0.2).all(), "one anchor per 5 s grid point"
    assert not np.any((times > 20.1) & (times < 29.9)), "anchors missing in the other camera must be dropped"
    assert len(select_anchor_windows(own, [other], stride_s=5.0, tolerance_s=0.1, max_windows=3)) == 3
    # Unsorted input returns indices into the caller's order.
    shuffled = own[::-1].copy()
    picked = select_anchor_windows(shuffled, [], stride_s=5.0, tolerance_s=0.1, max_windows=0)
    assert np.allclose(np.sort(shuffled[picked]), np.sort(own[select_anchor_windows(own, [], stride_s=5.0,
                                                                                   tolerance_s=0.1,
                                                                                   max_windows=0)]))
    assert select_anchor_windows(own, [np.array([])], stride_s=5.0, tolerance_s=0.1, max_windows=0) == []


def test_gait_cycle_recovers_square_wave_period_and_ignores_standing():
    times = np.arange(0.0, 20.0, 1 / 400)
    contact = ((times % 0.8) < 0.5).astype(float)  # 0.8 s gait cycle, 62% stance
    speed = np.where(times < 10.0, 1.0, 0.0)  # walks, then stands

    periods = gait_cycle_seconds(times, contact, speed)

    assert len(periods) == 1
    assert periods[0] == pytest.approx(0.8, abs=0.03)
    assert gait_cycle_seconds(times, contact, np.zeros_like(times)) == []


def test_linear_probe_recovers_linear_state_and_is_seeded():
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(2048, 16, generator=generator)
    states = latents @ torch.randn(16, 40, generator=generator) + 5.0
    config = {"steps": 400, "batch_size": 256, "learning_rate": 0.01, "weight_decay": 0.0}

    probe, fit = fit_probe("linear", latents, states, config, seed=1, device=torch.device("cpu"))
    again, _ = fit_probe("linear", latents, states, config, seed=1, device=torch.device("cpu"))

    assert r2_and_pearson(probe(latents), states)["all"]["r2"] > 0.99
    assert fit["parameters"] == 16 * 40 + 40
    assert torch.equal(probe(latents), again(latents))


def test_mlp_probe_has_protocol_architecture():
    _, fit = fit_probe("mlp", torch.randn(64, 8), torch.randn(64, 40),
                       {"steps": 1, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 0.0},
                       seed=0, device=torch.device("cpu"))
    assert fit["parameters"] == (8 * 256 + 256) + (256 * 256 + 256) + (256 * 40 + 40)


def test_holm_and_mann_whitney_follow_protocol_rules():
    assert holm([0.01, None, 0.04, 0.03]) == pytest.approx([0.03, None, 0.06, 0.06])
    assert mann_whitney([1.0, 2.0], [3.0, 4.0, 5.0]) is None
    assert mann_whitney([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(0.1)


def _tiny_lewm():
    return LeWorldModel(image_size=16, patch_size=8, encoder_dim=16, encoder_depth=1, encoder_heads=2,
                        latent_dim=8, predictor_dim=16, predictor_depth=1, predictor_heads=2,
                        predictor_window=2, proprio_dim=33, action_dim=6, context_steps=2,
                        rollout_steps=1, sigreg_slices=8).eval()


def _tiny_baseline():
    return JEPAWorldModel(visual_dim=8, proprio_dim=33, proprio_embed_dim=4, action_dim=6,
                          tokens_per_frame=4, predictor_depth=1, predictor_heads=2,
                          context_steps=2, rollout_context=2).eval()


def test_protocol_latents_use_only_the_models_trailing_context():
    torch.manual_seed(0)
    context_frames, steps = 4, 3
    frames = context_frames + steps
    batch = {"images": torch.rand(2, frames, 2, 16, 16), "proprio": torch.randn(2, frames, 33),
             "actions": torch.randn(2, frames - 1, 6)}
    tokens = torch.randn(2, frames, 4, 8)
    for model, extra in ((_tiny_lewm(), {}), (_tiny_baseline(), {"visual_tokens": tokens})):
        out = protocol_latents(model, batch, context_frames=context_frames, steps=steps, **extra)
        assert out["encoded"].shape[:2] == (2, frames)
        assert out["predicted"].shape == (2, steps, out["encoded"].shape[-1])
        # Frame 0 is outside both models' 2-frame context: predictions must not move.
        changed = {key: value.clone() for key, value in batch.items()}
        changed["proprio"][:, 0] += 10.0
        moved = protocol_latents(model, changed, context_frames=context_frames, steps=steps, **extra)
        assert torch.allclose(moved["predicted"], out["predicted"], atol=1e-5)
        assert not torch.allclose(moved["encoded"][:, 0], out["encoded"][:, 0])
        assert "predicted" not in protocol_latents(model, batch, context_frames=context_frames, steps=0, **extra)


def test_baseline_protocol_rollout_matches_training_recurrence():
    torch.manual_seed(0)
    model = _tiny_baseline()
    tokens, proprio, actions = torch.randn(2, 4, 4, 8), torch.randn(2, 4, 33), torch.randn(2, 3, 6)
    out = protocol_latents(model, {"proprio": proprio, "actions": actions}, context_frames=2, steps=2,
                           visual_tokens=tokens)
    observations = model.encode_observation(tokens, proprio)
    first = model.predict_next(observations[:, :2], actions[:, 1])
    second = model.predict_next(torch.cat((observations[:, 1:2], first[:, None]), 1), actions[:, 2])
    pooled = lambda tokens: torch.cat([part.mean(-2) for part in model._normalized_slices(tokens)], -1)
    assert torch.allclose(out["predicted"][:, 0], pooled(first), atol=1e-5)
    assert torch.allclose(out["predicted"][:, 1], pooled(second), atol=1e-5)


def test_state_vectors_can_anchor_base_position_at_rollout_start():
    states = [
        {"orientation": np.array([0.0, 0.0, 0.0, 1.0]), "pose_pos": np.array([float(i), 0.0, 0.0]),
         "lin_vel": np.zeros(3), "ang_vel": np.zeros(3), "gravity": np.zeros(3),
         "joint_pos": np.zeros(12), "joint_vel": np.zeros(12), "contacts": np.zeros(4)}
        for i in range(4)
    ]
    base = state_vectors(states, origin_tick=2)[:, STATE_LAYOUT["base_pos"]]
    assert np.allclose(base[:, 0], [-2.0, -1.0, 0.0, 1.0])
    assert np.allclose(state_vectors(states)[:, 0], [0.0, 1.0, 2.0, 3.0])


def _fake_eval(group: str, seed: int, eval_missions=("m1",), value: float = 1.0) -> dict:
    components = [*STATE_LAYOUT, "all", "all_excl_base_pos"]
    curve = {str(k): {"model": dict.fromkeys(components, value * k),
                      "persistence": dict.fromkeys(components, 2.0 * k),
                      "encoded_floor": dict.fromkeys(components, 0.1)} for k in (1, 5)}
    quality = {name: {"r2": 0.5, "pearson": 0.7} for name in [*STATE_LAYOUT, "all"]}
    return {
        "eval_config": "protocol-v1", "horizons": [1, 5],
        "model": {"group": group, "seed": seed, "type": "lewm", "epoch": 3},
        "data": {"eval_missions": list(eval_missions), "probe_missions": ["p1"], "context_frames": 7,
                 "match_observations": [], "tick_hz": 5.0},
        "probes": {"mlp": {"eps_k": curve, "quality": quality}},
        "compute": {"parameters_M": {"inference_total": 1.0}, "rollout_throughput_fps": 100.0},
        "gait_cycle": {"seconds": 0.8},
        "not_run": {"EV2_action_sensitivity": "needs sim"},
    }


def test_report_aggregates_seeds_and_flags_incomparable_runs():
    runs = [_fake_eval("lewm", s, value=1.0 + s / 10) for s in range(3)]
    runs += [_fake_eval("baseline", s, value=3.0 + s / 10) for s in range(3)]

    text = build_report(runs)

    assert "Not comparable" not in text
    assert "(n=3)" in text and "Holm" in text and "EV2_action_sensitivity" in text
    assert "≈ 4 ticks" in text
    assert comparability_problems(runs + [_fake_eval("lewm", 9, eval_missions=("other",))])


def test_training_compute_drops_restart_gaps(tmp_path):
    from quadwm.evaluation.protocol import training_compute

    records = [{"step": i, "timestamp": 1000.0 + 10 * i, "loss": 1.0} for i in range(4)]
    records += [{"eval/x": 1.0, "step": 3, "timestamp": 5000.0}]  # eval rows are not training time
    records += [{"step": 4, "timestamp": 9000.0, "loss": 1.0}, {"step": 5, "timestamp": 9010.0, "loss": 1.0}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (tmp_path / "run_metadata.json").write_text(json.dumps({"world_size": 2, "gpu_name": "MI300X"}),
                                                encoding="utf-8")

    compute = training_compute(tmp_path)

    assert compute["gpu_hours_approx"] == pytest.approx(40.0 * 2 / 3600)
    assert compute["gpu_name"] == "MI300X"


def test_eval_refuses_runs_without_a_probe_split(tmp_path):
    from quadwm.evaluation.protocol import _load_splits

    (tmp_path / "splits.json").write_text(json.dumps({"train": ["a"], "eval": ["b"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="adr/0001"):
        _load_splits(tmp_path, {})
    (tmp_path / "splits.json").write_text(json.dumps({"train": ["a"], "eval": ["b"], "probe": []}),
                                          encoding="utf-8")
    with pytest.raises(ValueError, match="empty probe split"):
        _load_splits(tmp_path, {})
    assert _load_splits(tmp_path, {"probe_split_fallback": "eval"})["probe"] == ["b"]

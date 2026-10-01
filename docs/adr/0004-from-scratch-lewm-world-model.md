# ADR 0004 — From-scratch Track 1 world model (LeWM-style)

- Status: proposed (needs E1.1 collapse gate on real data)
- Date: 2026-09-28

## Decision

`model.type: lewm` (`models/lewm.py`, `configs/jepa-wm/lewm_depth.yaml`)
trains a depth + proprio encoder and an action-conditioned predictor end to
end from random init. It runs through the same `quadwm train` pipeline;
no V-JEPA checkpoint or token cache is involved.

| Part | Choice | Source |
|---|---|---|
| Depth tokenizer | clip to range, valid-pixel area downsample to 64², symlog, 8×8 patches | Track 1 §2.1 (patch 8 not 16, see below) |
| Encoder | ViT-Tiny (192-d, 12 layers) over `[CLS, proprio, patches]` → CLS → 192-d latent | Track 1 §2.1 fusion; LeWM scale |
| Predictor | 6-layer AdaLN + RoPE, frame-causal, 4-frame window, one token per frame | Track 1 §2.3; reuses `AdaLNBlock` |
| Loss | teacher-forced next-latent MSE + open-loop rollout MSE + 0.1 · SIGReg | LeWM |
| Grounding | PSG state + transition heads, off by default (E1.3 arms) | Track 1 §2.4, §3.2–3.3 |

## Departures from the Track 1 recipe

- **SIGReg instead of EMA target + variance/covariance regularizer (§2.2, §3.4).**
  LeWM shows an end-to-end pixel JEPA trains stably with no EMA and no
  stop-gradient when SIGReg is the anti-collapse term, with a single
  weight to tune. Fewer moving parts than EMA schedule + two reg terms.
  E1.1 must confirm it on this data (log `effective_rank`, `latent_std`).
- **Patch 8 (64 tokens) instead of 16 (16 tokens).** Terrain geometry is
  the point of the depth channel; 16 tokens per frame is very coarse. Still
  cheap at one latent per frame.
- **Proprio is the current tick only**, not a 10-frame MLP history; the
  predictor's 4-frame window supplies history.
- **Invalid depth is an explicit validity channel** rather than a learned
  mask token per patch; equivalent information, less code.
- **Grounding heads default to off** so the default run is the E1.3 base
  arm. With proprio as an encoder input, the state head is close to an
  autoencoding target; the transition head is the more informative one.
- **Transition horizons {1, 4}** instead of {1, 4, 12}: a 12-tick horizon
  does not fit the 8-tick sample.

## From PiJEPA

PiJEPA's world model uses a frozen DINOv2/V-JEPA-2 encoder (jepa-wms), so
only its training schedule transfers: a rollout-steps curriculum (1 → 3 → 7)
by resuming with a larger `rollout_steps`. Its planner (policy-guided MPPI)
is evaluation and out of scope here.

## Open

- Depth camera frame rate and resolution on GrandTour are not yet checked
  against the 5 Hz tick (`max_tick_error_s` = 60 ms).
- SIGReg is computed per rank; per-GPU batch 128 keeps the statistic
  meaningful. Revisit if the batch shrinks.

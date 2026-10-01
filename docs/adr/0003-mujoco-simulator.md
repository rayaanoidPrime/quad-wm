# ADR 0003 — MuJoCo as the first simulator, behind a neutral contract

- Status: accepted
- Date: 2026-09-28

## Context

The Track 1 recipes name Isaac Lab. Isaac Sim officially supports NVIDIA
RTX GPUs; the cluster has AMD MI300X, so Isaac Lab camera rendering is not
a realistic target there. MISSION.md already lists MuJoCo as the first
candidate.

## Decision

- `quadwm.sim.base` defines the contract (`Simulator`, `TerrainSpec`,
  `DynamicsSpec`, `SimState`). Observations use GrandTour's vocabulary
  (`proprio` [33] plus the `sample_state` keys), so `state_vectors` and the
  probes consume sim and real data identically. Actions are GrandTour's:
  `12 * f` joint-position targets per 5 Hz tick.
- `MujocoSimulator` builds each episode with `MjSpec`: robot MJCF +
  terrain heightfield (`sim/terrain.py`) + front camera. Latency is a FIFO
  at physics-step resolution; `get_state`/`set_state` include that FIFO so
  EV2 replays are exact (verified bit-identical).
- **Robot is a stand-in.** MuJoCo Menagerie ships ANYmal B/C, not D.
  ANYmal C has the same joint names/order as GrandTour's D but different
  masses, geometry and actuators. Sim-to-real numbers (Δ_s2r) are not
  meaningful until an ANYmal D model passes a compatibility test.

## Open items (not solved here)

- ANYmal D MJCF (convert ANYbotics' description) and its compatibility test.
- A locomotion controller. `quadwm sim-smoke` only holds the standing pose;
  rollouts for EV1-sim/EV2/EV3/EV4 need a walking policy.
- Camera matching: GrandTour's front camera is fisheye on the Boxi payload;
  MuJoCo renders pinhole. Mount pose and FoV in the config are placeholders.
- Current terrains are left-right symmetric except `rough`, so `mirrored`
  is only meaningful once unilateral terrains exist (EV6).

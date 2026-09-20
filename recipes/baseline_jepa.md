# Track 1 — JEPA World Model for Quadruped Locomotion

- **Model:** Action-conditioned JEPA (ViT encoder, AdaLN predictor, multi-horizon prediction) with PSG-JEPA physical state grounding
- **Observation:** RGB
- **Conditioning:** Proprioception-derived: commanded actions plus a short proprioceptive history, injected through AdaLN
- **Platform:** ANYmal D
- **Data:** GrandTour (self-supervised pretraining and real held-out evaluation) plus Isaac Lab rollouts from a fixed controller
- **RL:** None in world-model training. Downstream RL on the frozen latent is part of the evaluation suite (E1.7).

---

## 0. Evidence status of each component

Read this before committing compute. Every method component below is a preprint or a non-peer-reviewed report. None has external replication on legged locomotion.

| Component | Source | Status | Risk |
|---|---|---|---|
| PSG-JEPA grounding | arXiv 2608.06799 (Yan et al., HKUST-GZ) | Preprint, Aug 2026. Evaluated on manipulation/navigation, not locomotion. | Medium |
| Multi-horizon prediction covering a gait cycle | Ground-JEPA, Research Square rs-10511499 | Not peer-reviewed, not on arXiv. Sim-only (DMC). | High |
| JEPA-WM design choices (AdaLN action conditioning, sizing, frozen-encoder comparison) | arXiv 2512.24497 (Terver et al., Meta FAIR + Inria) | OpenReview record, code released. Strongest evidence tier in this track. | Low |
| Physics prober pattern | SkyJEPA, arXiv 2606.23444 | Preprint. Quadrotor, not legged. Converges with Ground-JEPA. | Medium |
| GrandTour dataset | arXiv 2602.18164 | Public dataset paper; data on Hugging Face. 49 missions, >5 h. | Low |

**Interpretation.** This track is a synthesis, not a reproduction. The JEPA-WM ablation study (2512.24497) provides the design scaffold. The grounding heads and the multi-horizon set are ablation arms with explicit on/off switches, not fixed architecture.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $\Delta t$ | World-model tick. All streams are resampled to this rate (start with $0.1$ s). |
| $I_t \in \mathbb{R}^{R \times R \times 3}$ | RGB frame at tick $t$ (start with $R = 128$) |
| $s_t^{\text{prop}}$ | Proprioceptive state: base linear/angular velocity, gravity vector in base frame, joint positions $q_t$, joint velocities $\dot q_t$ |
| $q_t \in \mathbb{R}^{12}$ | Joint angle vector (ANYmal D: 12 actuated joints) |
| $a_t$ | Action over tick $t \to t+1$. Commanded joint-position targets $\in \mathbb{R}^{12}$ if logged in both data sources; otherwise commanded base twist $\in \mathbb{R}^{3}$ (§2.3). |
| $z_t \in \mathbb{R}^{D}$ | Latent from the context encoder $E_\theta$, $D = 192$ |
| $Z_t$ | Context window of latents $(z_{t-m+1}, \dots, z_t)$, start with $m = 3$ |
| $\bar z_t$ | Latent from the EMA target encoder $E_{\bar\theta}$ |
| $c_t$ | Proprioceptive history $(s^{\text{prop}}_{t-N+1}, \dots, s^{\text{prop}}_t)$, start with $N = 5$ |
| $\hat z_{t+k}$ | Predicted latent at horizon $k$ from the predictor $P_\theta$ |
| $\text{sg}[\cdot]$ | Stop-gradient operator |
| $\mathcal{H}$ | Set of prediction horizons used in training |
| $T_{\text{gait}}$, $H_{\text{gait}}$ | Gait period (s) and its length in ticks, $H_{\text{gait}} = \operatorname{round}(T_{\text{gait}} / \Delta t)$ |

---

## 2. Data

### 2.1 Sources

| Source | Role | Notes |
|---|---|---|
| GrandTour (ANYmal D) | (a) Self-supervised pretraining corpus, (b) real-world held-out evaluation | Time-synchronized RGB cameras, joint encoders (position, velocity, torque), commanded motion. No reward and no privileged terrain labels. |
| Isaac Lab rollouts | Mixed fine-tuning on simulated RGB; privileged-label probes; environment for the downstream RL evaluation (E1.7) | Rollouts from a fixed pretrained ANYmal D locomotion controller, trained outside this track. Randomize velocity commands, add action noise and random pushes, randomize terrain, friction, mass, and lighting/textures. Log RGB, proprioception, commands, terrain height, friction, and contacts. |

### 2.2 Verify before starting

Confirm directly from the dataset documentation; do not assume:

- Which RGB camera(s) to use, their frame rate, resolution, and mounting pose relative to the base. Start with a single forward-facing camera; additional cameras can serve as extra pretraining views later.
- Time synchronization between RGB and proprioception, and the proprioceptive logging rate.
- Which command signals are logged: joint-position targets, base-twist command, or both.
- Configure the Isaac Lab camera to match the chosen GrandTour camera's field of view and mounting pose as closely as possible.

### 2.3 Action definition

The action must have the same definition in both data sources. If GrandTour logs joint-position targets, $a_t \in \mathbb{R}^{12}$ is the commanded target resampled to the tick. If it logs only the base-twist command, $a_t = (v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}})$ in both sources, and the joint-level dynamics are carried by the proprioceptive history $c_t$. Simulation logs both, so the choice is set by what GrandTour provides. Decide in E1.0 and hold fixed for every experiment.

### 2.4 Split protocol

Partition GrandTour by *mission*, not by frame. Consecutive frames within a mission are near-duplicates, so frame-level splits leak. Reserve at least 20% of missions (at least 10 of the 49), spanning all terrain and illumination conditions present, as a held-out set that is never touched during training, probe fitting, or hyperparameter selection. In simulation, hold out terrain seeds and terrain types.

### 2.5 Augmentation

Photometric augmentation (color jitter, blur, noise, exposure) and small random-resized crops, with parameters shared across every frame in a sequence window. No horizontal flips.

---

## 3. Architecture

### 3.1 Encoder

- **Tokenizer and backbone:** ViT on RGB. Resize to $128 \times 128$, patch size $16$ (64 patch tokens plus a `[CLS]` token), pooled to a single latent $z_t \in \mathbb{R}^{192}$. Each frame is encoded independently.
- **Input is RGB only, by design.** Proprioception is not an encoder input. It enters the predictor as conditioning (§3.3) and the grounding heads as targets (§3.4). If proprioception were fed to the encoder, the latent could copy it and the grounding heads would be trivially satisfiable.
- **Sizing.** Terver et al. (2512.24497) found that increasing encoder size or predictor depth did not improve performance on simulated benchmarks and could hurt it. LeWM (arXiv 2603.19312) operates at roughly 15M parameters with a 192-dimensional per-frame latent. Start at that scale (for example a ViT-Tiny-scale encoder with width 192). Do not scale up before the ablations in §6 justify it.

### 3.2 Target encoder

Standard JEPA asymmetry. $E_{\bar\theta}$ shares architecture with $E_\theta$ and is updated by exponential moving average:

$$\bar\theta \leftarrow \tau \bar\theta + (1 - \tau)\theta$$

with $\tau$ ramped from $0.996$ to $1.0$ over training. Gradients never flow through $E_{\bar\theta}$.

### 3.3 Predictor

Transformer over the context window $Z_t$ that reads out $\hat z_{t+k}$ for each horizon $k \in \mathcal{H}$ in one pass:

$$\hat z_{t+k} = P_\theta\big(Z_t;\; a_{t:t+k-1},\, c_t,\, k\big)$$

Conditioning enters through AdaLN (adaptive layer norm), which Terver et al. identify as the effective conditioning mechanism in this family. The conditioning vector is produced by an MLP over three embeddings: the action sequence $a_{t:t+k-1}$ (padded and masked to the maximum horizon, encoded by a small temporal encoder), the proprioceptive history $c_t$, and the horizon $k$.

### 3.4 Grounding heads (training-only, PSG-JEPA)

Two lightweight MLP heads, discarded at inference so the deployed model's compute cost is unchanged:

- **State head** $h_\psi: Z_t \mapsto \hat s_t^{\text{prop}}$. It reads the context window rather than a single latent, because base and joint velocities are not recoverable from one frame.
- **Transition head** $g_\phi: (z_t, z_{t+h}) \mapsto \widehat{\Delta q}_{t,h}$

**Observability.** An egocentric RGB camera observes robot state only partially. Orientation (gravity direction) and base motion are largely inferable from the image stream; joint positions and velocities are recoverable mainly through their correlation with gait phase and ego-motion. Report probe results per component of $s^{\text{prop}}$, not as a single aggregate.

---

## 4. Training objectives

### 4.1 Forward prediction (base JEPA loss)

$$\mathcal{L}_{\text{pred}} = \frac{1}{|\mathcal{H}|}\sum_{k \in \mathcal{H}} \frac{1}{T}\sum_{t} \Big\| P_\theta\big(Z_t;\, a_{t:t+k-1},\, c_t,\, k\big) - \text{sg}\big[E_{\bar\theta}(I_{t+k})\big] \Big\|_2^2$$

Default horizon set: $\mathcal{H} = \{1, 4, H_{\text{gait}}\}$.

### 4.2 State grounding (PSG-JEPA, objective 1)

$$\mathcal{L}_{\text{state}} = \frac{1}{T}\sum_{t} \big\| h_\psi(Z_t) - \tilde s_t^{\text{prop}} \big\|_2^2$$

where $\tilde s^{\text{prop}}$ is $s^{\text{prop}}$ standardized per dimension with training-set statistics, so no single unit scale dominates.

### 4.3 Transition grounding (PSG-JEPA, objective 2)

$$\mathcal{L}_{\text{trans}} = \frac{1}{|\mathcal{H}_g|}\sum_{h \in \mathcal{H}_g} \frac{1}{T}\sum_{t} \big\| g_\phi(z_t, z_{t+h}) - \widetilde{(q_{t+h} - q_t)} \big\|_2^2$$

with $\mathcal{H}_g = \mathcal{H}$ and the joint-angle change standardized per horizon. The multi-horizon set supervises both fast joint dynamics and a full gait cycle.

### 4.4 Anti-collapse regularization

Non-contrastive joint-embedding objectives admit collapsed solutions. EMA plus stop-gradient alone is not a guarantee. Add an explicit variance/covariance term over the batch of latents:

$$\mathcal{L}_{\text{reg}} = \frac{1}{D}\sum_{j=1}^{D} \max\big(0,\; \gamma - \sqrt{\text{Var}(z^{(j)}) + \epsilon}\big) \;+\; \frac{1}{D}\sum_{i \neq j} \big[\text{Cov}(z)\big]_{i,j}^2$$

with $\gamma = 1$, $\epsilon = 10^{-4}$.

### 4.5 Total

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_s \mathcal{L}_{\text{state}} + \lambda_\tau \mathcal{L}_{\text{trans}} + \lambda_r \mathcal{L}_{\text{reg}}$$

**Starting values:** $\lambda_s = 1.0$, $\lambda_\tau = 1.0$, $\lambda_r = 0.1$. These are unvalidated for locomotion. Sweep them in E1.3.

---

## 5. Training procedure

There is no RL loss and no reward signal anywhere in world-model training.

- **Stage A — GrandTour pretraining.** Train on train-split GrandTour missions with all four loss terms. Proprioception is available in the logs, so the grounding heads apply. Run the collapse diagnostics (E1.1) throughout.
- **Stage B — Mixed fine-tuning.** Continue on batches mixed from GrandTour and Isaac Lab (start 50/50) so simulated RGB is in-distribution for the downstream RL evaluation without discarding the real-data prior. Privileged simulation labels (terrain height, friction, contacts) are used only for probes and evaluation, never in a training loss.
- **Freeze** the encoder after Stage B. Every evaluation in §6 uses the frozen model.

---

## 6. Experiment plan

Run these in order. Each stage gates the next.

### E1.0 — Infrastructure and data validation (no science)
- Confirm GrandTour RGB and proprioception load and are time-aligned; settle the items in §2.2 and the action definition in §2.3.
- Confirm Isaac Lab ANYmal D renders RGB and that the fixed controller produces stable rollouts. Measure simulation throughput (FPS) with camera rendering on before planning any experiment budget.
- Measure $T_{\text{gait}}$ on the simulation controller, check it against GrandTour logs, and set $H_{\text{gait}}$.
- **Gate:** aligned data, defined action, known FPS, known $H_{\text{gait}}$.

### E1.1 — Collapse check (highest-priority failure mode)
- Train encoder + predictor with $\mathcal{L}_{\text{pred}}$ only (grounding and regularization off).
- Log every 500 steps: latent variance per dimension, effective rank of the latent covariance matrix, and the ratio $\|\hat z_{t+k} - \bar z_{t+k}\| / \|\bar z_{t+k}\|$.
- **Failure signature:** effective rank collapsing toward 1, or per-dimension variance approaching zero. This is silent: the loss will look like it is decreasing.
- Then repeat with $\mathcal{L}_{\text{reg}}$ enabled.
- **Gate:** effective rank stable above a chosen floor (for example 50% of $D$) for 50k steps.

### E1.2 — Encoder ablation
The Terver et al. finding that a strong frozen DINO-class encoder preserves planning-relevant physical detail was obtained on RGB benchmarks, so RGB is the native setting here. Whether it holds for outdoor egocentric legged-robot video, with only about 5 h of real data, is open.

| Arm | Encoder |
|---|---|
| A | End-to-end trained ViT from §3.1 (default) |
| B | Frozen DINOv2 ViT-S/14 at $224 \times 224$ (features cached), predictor trained on top |
| C | DINOv2-initialized ViT fine-tuned end-to-end with a reduced encoder learning rate |

- **Metrics:** probe $R^2$ on physical state (per component), plus $\mathcal{L}_{\text{pred}}$ at convergence and compute cost.
- **Expected outcome:** uncertain. With limited real data, arm A may overfit and arms B/C may win. A null or reversed result is publishable on its own.

### E1.3 — Grounding ablation (the core PSG-JEPA test)

| Arm | $\lambda_s$ | $\lambda_\tau$ |
|---|---|---|
| Base | 0 | 0 |
| State only | 1.0 | 0 |
| Transition only | 0 | 1.0 |
| Full PSG | 1.0 | 1.0 |

- Also sweep $\lambda_s, \lambda_\tau \in \{0.1, 1.0, 10.0\}$ on the full arm.
- Also compare state-head targets: the full $s^{\text{prop}}$ versus the subset that egocentric RGB can plausibly identify (gravity direction and base twist).
- **Metrics:** probe $R^2$ / Pearson $r$ on proprioceptive state and on joint-angle change, per component, on held-out GrandTour missions and held-out simulated terrains; downstream RL success (E1.7) for at least Base versus Full PSG.
- **Why this is the headline experiment:** PSG-JEPA's identifiability claim has never been tested on legged locomotion, and here the observation is a partially observing egocentric camera. The Base-versus-Full delta answers whether the robot-centric identifiability gap is the bottleneck. A null result is a real finding.

### E1.4 — Prediction horizon ablation
- Sweep $\mathcal{H} \in \{\{1\},\ \{1,4\},\ \{1,4,H_{\text{gait}}\},\ \{1,4,H_{\text{gait}},2H_{\text{gait}}\}\}$.
- Ground-JEPA reports that extending the horizon to cover a full gait cycle (12 steps in its setup) surpassed reward-grounded performance on quadruped-walk (510 ± 68 vs 450 ± 234, $p < 0.05$).
- **Caveat to carry into interpretation:** that baseline's standard deviation (±234) overlaps the claimed improvement substantially, and the setting is sim-only. Treat a gait-cycle horizon as a hypothesis under test. Report your own variance across at least 3 seeds.
- **Metrics:** probe $R^2$ and open-loop prediction error broken down by horizon.

### E1.5 — Pretraining ablation

| Arm | Training data |
|---|---|
| A | Isaac Lab only, from scratch |
| B | GrandTour only |
| C | GrandTour pretraining, then mixed fine-tuning (default, §5) |

- **Metrics:** probe $R^2$ on held-out GrandTour missions and on held-out simulated terrains. This measures both the benefit of real-data pretraining and the residual sim-to-real appearance gap.

### E1.6 — Open-loop real-data evaluation
- Freeze the trained world model. On held-out GrandTour missions, run open-loop prediction from logged RGB and logged actions.
- Evaluate both the direct multi-horizon readout and chained $k = 1$ predictions. The two are not guaranteed to agree.
- Decode predicted latents with probes fit on training missions only, and compare predicted physical state against logged state estimates, by horizon and by terrain type.
- Include a persistence baseline, $\hat z_{t+k} = z_t$, so the reported skill is measured against "nothing changes".
- **This is the differentiator.** Very few world-model papers evaluate prediction against real logged data at this scale. Report it regardless of whether it flatters the model.

### E1.7 — Downstream RL evaluation (evaluation suite)
The frozen world model is evaluated as a representation for control. RL happens here, not in training.

- **Setup:** Isaac Lab ANYmal D, PPO with an asymmetric actor-critic (privileged critic). The actor receives the frozen $z_t$ (or $Z_t$) plus a proprioceptive history. The encoder stays frozen.
- **Reward and curriculum:** Isaac Lab ANYmal rough-terrain defaults, following Rudin et al. (CoRL 2022), held identical across every variant and across tracks.
- **Comparison arms:**
  1. Proprioception-only (blind) policy
  2. End-to-end RGB encoder trained by RL alone
  3. Frozen DINOv2 features
  4. Frozen world-model latents (Base, Full PSG, best horizon set from E1.4)
- **Metrics:** terrain level reached, success rate at a fixed RL budget, and steps to a reward threshold. At least 3 seeds.
- **Reading the result:** if frozen world-model latents do not beat arms 1–3, that is a finding about what the latent contains; check the per-component probe deficits from E1.3 for the reason.

---

## 7. Known risks and mitigations

| Risk | Signature | Mitigation |
|---|---|---|
| Representation collapse | Latent effective rank drops; loss still decreasing | E1.1 gate; keep $\mathcal{L}_{\text{reg}}$ on |
| Partial observability of robot state from RGB | State-head loss plateaus; low probe $R^2$ on joint velocities | Window-based state head; per-component reporting; state-target subset arm in E1.3 |
| Limited real data (about 5 h) | Large train/held-out mission gap | Augmentation; extra cameras as views; arms B/C in E1.2; monitor held-out missions throughout |
| Sim-to-real appearance gap | Probe $R^2$ good in sim, poor on GrandTour | GrandTour pretraining; mixed fine-tuning; lighting/texture randomization; camera matching (§2.2) |
| Action mismatch between sources | Different action distributions or definitions for sim vs real | Single action definition fixed in E1.0; report metrics per source |
| Tick or synchronization error | Poor probe $R^2$ even at $k = 1$ | Verify alignment in E1.0; resample carefully to $\Delta t$ |
| Mission leakage | Held-out results inflated | Mission-level split (§2.4) |
| Frozen latent lacks control-relevant information | Arm 4 trails arms 2–3 in E1.7 | Diagnose with per-component probes; consider fine-tuning extensions (§8) |
| Preprint drift | PSG-JEPA or Ground-JEPA revise method or results | Re-check versions and released code before implementing from paper text |

---

## 8. Deferred (not part of this recipe)

- Additional modalities: depth and LiDAR, including the PE-JEPA point-cloud encoder from JEPLO (arXiv 2609.15770).
- Joint world-model and RL training with a gradient-isolation schedule (JEPLO's concurrent teacher–student design).
- Mirror-symmetry data augmentation.

---

## 9. References

- PSG-JEPA: arXiv 2608.06799 — Yan et al. *Is Forward Prediction Enough? Physical State Grounding for JEPA World Models.*
- JEPA-WM design study: arXiv 2512.24497 — Terver et al. (Meta FAIR, Inria). *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?*
- Ground-JEPA: Research Square rs-10511499. *Learning Physically Grounded Latent World Models for Zero-Shot Dynamics Generalization of Legged Robots.*
- SkyJEPA: arXiv 2606.23444. *Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors.*
- LeWM: arXiv 2603.19312 — Maes et al. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.*
- GrandTour: arXiv 2602.18164. *GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation.*
- DINOv2: Oquab et al., 2023. V-JEPA 2: Assran et al., 2025. DINO-WM: Zhou et al. PLDM: Sobal et al., 2025.
- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2022, pp. 91–100.
- JEPLO (deferred, §8): arXiv 2609.15770 — Yuan, Qiu, Cao, Cao, Li. *Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion.*
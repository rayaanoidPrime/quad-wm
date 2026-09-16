# Track 1 — JEPA World Model for Quadruped Locomotion

**Base system:** JEPLO (PE-JEPA + concurrent teacher–student RL)
**Added objective:** PSG-JEPA physical state grounding
**Primary sensor:** depth images (LiDAR deferred to stretch goal)
**Platform:** ANYmal D
**Simulator:** Isaac Lab

---

## 0. Evidence status of each component

Read this before committing compute. Every component below is a preprint. None has external replication.

| Component | Source | Status | Risk |
|---|---|---|---|
| JEPLO / PE-JEPA | arXiv 2609.15770 (Yuan, Qiu, Cao, Cao, Li) | Preprint, Sept 2026. No quantitative real-robot comparison table published. | High |
| PSG-JEPA grounding | arXiv 2608.06799 (Yan et al., HKUST-GZ) | Preprint, Aug 2026. Evaluated on manipulation/navigation, not locomotion. | Medium |
| H=12 gait-cycle horizon | Ground-JEPA, Research Square rs-10511499 | Not peer-reviewed, not on arXiv. Sim-only (DMC). | High |
| JEPA-WM design choices | arXiv 2512.24497 (Terver et al., Meta FAIR + Inria) | OpenReview record, code released. Strongest evidence tier in this track. | Low |
| Physics prober pattern | SkyJEPA, arXiv 2606.23444 | Preprint. Quadrotor, not legged. Converges with Ground-JEPA. | Medium |

**Interpretation.** This track is a synthesis, not a reproduction. The JEPLO system design is the scaffold; the design choices inside it come from the higher-evidence ablation study (2512.24497). Treat the H=12 horizon and the grounding heads as ablation arms with explicit on/off switches, not as fixed architecture.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $o_t$ | Observation at time $t$: depth image $d_t \in \mathbb{R}^{H \times W}$ concatenated with proprioceptive history |
| $s_t^{\text{prop}}$ | Proprioceptive state vector (base linear/angular velocity, gravity vector in base frame, joint positions $q_t$, joint velocities $\dot q_t$) |
| $q_t \in \mathbb{R}^{12}$ | Joint angle vector (ANYmal D: 12 actuated joints) |
| $a_t \in \mathbb{R}^{12}$ | Action — target joint positions fed to the PD controller |
| $z_t$ | Latent produced by the context encoder $E_\theta$ |
| $\bar z_t$ | Latent produced by the EMA target encoder $E_{\bar\theta}$ |
| $\hat z_{t+k}$ | Predicted latent at horizon $k$ from the predictor $P_\theta$ |
| $\text{sg}[\cdot]$ | Stop-gradient operator |
| $H$ | Set of prediction horizons used in training |

---

## 2. Architecture

### 2.1 Encoder (this is the component that changes for depth)

JEPLO's published encoder operates on LiDAR point clouds. Depth images require a different tokenizer. The following is the substitution:

- **Depth tokenizer:** patch-based ViT encoder. Split $d_t$ into non-overlapping patches of size $p \times p$ (start with $p = 16$ on a $64 \times 64$ downsampled depth map, giving 16 tokens per frame).
- **Depth preprocessing:** clip to sensor range $[d_{\min}, d_{\max}]$, then apply symlog normalization to compress the dynamic range:
  $$\text{symlog}(x) = \text{sign}(x)\,\ln(|x| + 1)$$
  Invalid / no-return pixels get a learned mask token rather than a sentinel value.
- **Proprioceptive branch:** MLP encoder over a history window of $N$ past proprioceptive frames (start with $N = 10$). Concatenate the resulting embedding with the depth tokens before the transformer.
- **Fusion:** single transformer over `[depth patch tokens] + [proprio token]`, output pooled to a single latent vector $z_t$.

**Sizing.** Terver et al. (2512.24497) found that increasing encoder size or predictor depth did not improve performance on simulated benchmarks and could hurt it. LeWM (arXiv 2603.19312) operates at roughly 15M parameters with a 192-dimensional per-frame latent. Start at that scale. Do not scale up before the ablations in §6 justify it.

### 2.2 Target encoder

Standard JEPA asymmetry. The target encoder $E_{\bar\theta}$ shares architecture with $E_\theta$ but is updated by exponential moving average:

$$\bar\theta \leftarrow \tau \bar\theta + (1 - \tau)\theta$$

with $\tau$ ramped from $0.996$ to $1.0$ over training. Gradients never flow through $E_{\bar\theta}$.

### 2.3 Predictor

Action-conditioned transformer predictor $P_\theta$. Given $z_t$ and an action sequence $a_{t:t+k-1}$, predict $\hat z_{t+k}$. Condition on actions via AdaLN (adaptive layer norm), which Terver et al. identify as the effective conditioning mechanism in this family.

### 2.4 Grounding heads (training-only — PSG-JEPA)

Two lightweight MLP heads, discarded at inference so the deployed model's compute cost is unchanged:

- **State head** $h_\psi: z_t \mapsto \hat s_t^{\text{prop}}$
- **Transition head** $g_\phi: (z_t, z_{t+h}) \mapsto \widehat{\Delta q}_{t,h}$

---

## 3. Training objectives

### 3.1 Forward prediction (the base JEPA loss)

$$\mathcal{L}_{\text{pred}} = \frac{1}{|H|}\sum_{k \in H} \frac{1}{T}\sum_{t} \Big\| P_\theta\big(E_\theta(o_t),\, a_{t:t+k-1}\big) - \text{sg}\big[E_{\bar\theta}(o_{t+k})\big] \Big\|_2^2$$

### 3.2 State grounding (PSG-JEPA, objective 1)

Grounds individual latents in robot proprioceptive state:

$$\mathcal{L}_{\text{state}} = \frac{1}{T}\sum_{t} \big\| h_\psi(z_t) - s_t^{\text{prop}} \big\|_2^2$$

### 3.3 Transition grounding (PSG-JEPA, objective 2)

Grounds latent *pairs* in multi-horizon joint-angle changes:

$$\mathcal{L}_{\text{trans}} = \frac{1}{|H_g|}\sum_{h \in H_g} \frac{1}{T}\sum_{t} \big\| g_\phi(z_t, z_{t+h}) - (q_{t+h} - q_t) \big\|_2^2$$

Use a multi-horizon set, e.g. $H_g = \{1, 4, 12\}$, so the objective supervises both fast joint dynamics and a full gait cycle.

### 3.4 Anti-collapse regularization

Non-contrastive joint-embedding objectives admit collapsed solutions. EMA + stop-gradient alone is not a guarantee. Add an explicit variance/covariance term:

$$\mathcal{L}_{\text{reg}} = \frac{1}{D}\sum_{j=1}^{D} \max\big(0,\; \gamma - \sqrt{\text{Var}(z^{(j)}) + \epsilon}\big) \;+\; \frac{1}{D}\sum_{i \neq j} \big[\text{Cov}(z)\big]_{i,j}^2$$

with $\gamma = 1$, $\epsilon = 10^{-4}$.

### 3.5 Total

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_s \mathcal{L}_{\text{state}} + \lambda_\tau \mathcal{L}_{\text{trans}} + \lambda_r \mathcal{L}_{\text{reg}}$$

**Starting values:** $\lambda_s = 1.0$, $\lambda_\tau = 1.0$, $\lambda_r = 0.1$. These are unvalidated for locomotion — sweep them in E1.3.

---

## 4. The RL loop (JEPLO's concurrent teacher–student)

JEPLO's contribution is that the world model and the policy train *concurrently*, not in a pretrain-then-distill sequence. This differs from the classical privileged-learning pipeline (RMA, Kumar et al. 2021) and from the depth-based teacher–student used in most parkour work.

- **Teacher:** privileged critic with access to ground-truth terrain geometry, friction coefficients, and base state from the simulator.
- **Student:** actor conditioned on $z_t$ (the JEPA latent) plus the proprioceptive history. No privileged access.
- **Concurrency:** on each update, run the world-model loss (§3.5) and the RL loss in the same optimizer step, with separate learning rates.
- **Gradient isolation:** do **not** let the policy gradient flow back into the encoder in the first phase. Attach a stop-gradient on $z_t$ at the policy input for the first $K$ iterations (start with $K = 5000$), then release it. This prevents the reward signal from collapsing the latent before the self-supervised objective has shaped it.

### 4.1 Reward terms

Base locomotion reward, following the stabilized recipe from Rudin et al. (CoRL 2022, pp. 91–100):

- **Linear velocity tracking:**
  $$r_{v_{xy}} = \exp\left(-\frac{\|v_{xy}^{\text{cmd}} - v_{xy}\|_2^2}{\sigma_v}\right), \quad \sigma_v = 0.25$$
- **Angular velocity tracking:**
  $$r_{\omega_z} = \exp\left(-\frac{(\omega_z^{\text{cmd}} - \omega_z)^2}{\sigma_\omega}\right), \quad \sigma_\omega = 0.25$$
- **Torque penalty:** $r_\tau = -\|\tau\|_2^2$
- **Joint acceleration penalty:** $r_{\ddot q} = -\|\ddot q\|_2^2$
- **Action rate penalty:** $r_{\Delta a} = -\|a_t - a_{t-1}\|_2^2$
- **Collision / base contact penalty:** large negative on undesired body contacts.

Weights follow the Isaac Lab ANYmal rough-terrain defaults as a starting point. Do not hand-tune these before the world model trains stably — reward tuning on top of an unstable latent wastes time.

### 4.2 Terrain curriculum

Progressive difficulty per Rudin et al.: flat → slopes → stairs → gaps → climbs. Advance a terrain tier when the policy exceeds a traversal threshold (e.g. 80% success over the last 100 episodes) on the current tier.

### 4.3 Symmetry data augmentation (optional, cheap)

Mittal et al. (IROS 2024) showed mirroring observations and actions as a PPO data-augmentation pass improves sample efficiency and gait quality. This is the weak form of SWAP's hard equivariance constraint. It costs almost nothing here and is worth enabling by default — flag it in ablation E1.5 so you can measure its contribution separately.

---

## 5. Data

| Source | Role | Notes |
|---|---|---|
| Isaac Lab rollouts | Primary training data for the RL loop and the world model | Only source with reward signal. Carries the entire RL loop. |
| GrandTour (ANYmal D) | (a) Self-supervised pretraining corpus for the encoder, (b) real-world open-loop evaluation set | No reward. Actions are only implicit in logged proprioceptive commands. Cannot train the RL loop. |

**Before starting:** verify the exact GrandTour sensor suite, frame rates, and per-mission duration from the dataset documentation. Do not assume depth camera availability, resolution, or synchronization with proprioception — confirm these directly, since the depth pipeline's input specification depends on them.

**Split protocol.** Partition GrandTour by *mission*, not by frame. Frame-level splits leak, because consecutive frames within a mission are near-duplicates. Reserve at least 20% of missions, spanning all terrain types present, as a held-out evaluation set that is never touched during training or probe fitting.

---

## 6. Experiment plan

Run these in order. Each stage gates the next.

### E1.0 — Infrastructure validation (no science)
- Confirm Isaac Lab ANYmal D environment runs with depth camera rendering enabled.
- Measure achievable simulation throughput (FPS) with depth rendering on. Depth rendering is the throughput bottleneck; record the number before planning any experiment budget.
- Verify GrandTour loads and that depth + proprioception streams are time-aligned.
- **Gate:** stable rollouts, known FPS.

### E1.1 — Collapse check (highest-priority failure mode)
- Train the encoder + predictor on Isaac Lab rollouts with $\mathcal{L}_{\text{pred}}$ only (grounding and regularization off).
- Log every 500 steps: latent variance per dimension, effective rank of the latent covariance matrix, and the ratio $\|\hat z_{t+k} - \bar z_{t+k}\| / \|\bar z_{t+k}\|$.
- **Failure signature:** effective rank collapsing toward 1, or per-dimension variance approaching zero. This is silent — the loss will look like it is decreasing.
- Then repeat with $\mathcal{L}_{\text{reg}}$ enabled.
- **Gate:** effective rank stable above a chosen floor (e.g. 50% of latent dimensionality) for 50k steps.

### E1.2 — Encoder ablation
Terver et al.'s finding that a strong frozen visual encoder (DINO-class) preserves planning-relevant physical detail, while some video-style encoders blur it, is a result about **RGB** encoders. It is not established for depth. Test it rather than assuming it.

| Arm | Encoder |
|---|---|
| A | End-to-end trained ViT on depth (default) |
| B | Frozen DINOv2 applied to depth-as-3-channel, predictor trained on top |
| C | Frozen DINOv2 on RGB (if available), depth ignored — reference point only |

- **Metric:** probe $R^2$ on physical state (see the shared evaluation protocol), plus $\mathcal{L}_{\text{pred}}$ at convergence.
- **Expected outcome:** uncertain, which is why this is an experiment. Depth lacks the texture statistics DINOv2 was trained on, so arm A may win. A null or reversed result here is publishable on its own.

### E1.3 — Grounding ablation (the core PSG-JEPA test)

| Arm | $\lambda_s$ | $\lambda_\tau$ |
|---|---|---|
| Base | 0 | 0 |
| State only | 1.0 | 0 |
| Transition only | 0 | 1.0 |
| Full PSG | 1.0 | 1.0 |

- Also sweep $\lambda_s, \lambda_\tau \in \{0.1, 1.0, 10.0\}$ on the full arm.
- **Metrics:** probe $R^2$ / Pearson $r$ on proprioceptive state and on joint-angle change; downstream policy success rate at fixed RL budget.
- **Why this is the headline experiment:** PSG-JEPA's identifiability claim has never been tested on legged locomotion. The base-vs-full delta answers directly whether the robot-centric identifiability gap is the bottleneck in a locomotion JEPA. A null result is a real finding.

### E1.4 — Prediction horizon ablation
- Sweep $H \in \{\{1\}, \{1,4\}, \{1,4,12\}, \{1,4,12,25\}\}$.
- Ground-JEPA reports that extending the horizon to cover a full gait cycle ($H = 12$) surpassed reward-grounded performance on quadruped-walk (510 ± 68 vs 450 ± 234, $p < 0.05$).
- **Caveat to carry into interpretation:** that baseline's standard deviation (±234) is wide enough to overlap the claimed improvement substantially. Treat $H = 12$ as a hypothesis under test, not a settled result. Report your own variance across at least 3 seeds.
- **ANYmal D gait-cycle note:** $H = 12$ corresponds to one gait cycle only at a specific control frequency. Compute the actual step count for one full ANYmal D gait cycle at your control rate and set the horizon from that, rather than copying 12 directly.

### E1.5 — Symmetry augmentation on/off
- Toggle the mirroring augmentation from §4.3.
- **Metric:** sample efficiency (steps to reach a fixed reward threshold), and mirrored-terrain transfer (see shared evaluation protocol, EV6).

### E1.6 — Gradient isolation schedule
- Sweep $K \in \{0, 1000, 5000, 20000\}$.
- $K = 0$ means the policy gradient reaches the encoder from step one. Include it — if it does not hurt, the schedule is unnecessary complexity.

### E1.7 — Real-data transfer (GrandTour)
- Freeze the trained world model. Run open-loop prediction on held-out GrandTour missions using logged depth observations and logged commanded actions.
- Compare predicted physical state against logged state estimates.
- **This is the differentiator.** Very few world-model papers evaluate prediction against real logged data at this scale. Report it regardless of whether it flatters the model.

### E1.8 — Stretch: LiDAR
Only after E1.1–E1.7 are complete and stable.
- Replace the depth ViT tokenizer with JEPLO's native PE-JEPA point-cloud encoder.
- Everything in §3 is unchanged — the grounding heads are modality-agnostic and supervise whatever latent the encoder produces.
- **Expected benefit:** JEPLO's stated advantage is that the latent retains task-relevant structure under occlusion, sparsity, and noise — the specific failure mode of raw LiDAR on legged platforms. Test this claim directly by injecting synthetic dropout/noise into the LiDAR stream and measuring probe degradation against the depth baseline.

---

## 7. Known risks and mitigations

| Risk | Signature | Mitigation |
|---|---|---|
| Representation collapse | Latent effective rank drops; loss still decreasing | E1.1 gate; keep $\mathcal{L}_{\text{reg}}$ on |
| Depth sim-to-real gap | Probe $R^2$ good in sim, poor on GrandTour | Domain randomization on depth noise, dropout, and range clipping in Isaac Lab |
| Policy gradient corrupts latent | Probe $R^2$ degrades as RL reward improves | Gradient isolation schedule (§4.4), E1.6 |
| Depth rendering throughput | Wall-clock per experiment explodes | Measure in E1.0; consider lower depth resolution or reduced camera update rate |
| JEPLO code diverges from paper | Reimplementation does not match | Check the ASIG-X repository for released code before reimplementing from paper text; a two-week-old preprint may revise its method section in v2 |

---

## 8. References

- JEPLO: arXiv 2609.15770 — Yuan, Qiu, Cao, Cao, Li. *Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion.*
- PSG-JEPA: arXiv 2608.06799 — Yan et al. *Is Forward Prediction Enough? Physical State Grounding for JEPA World Models.*
- JEPA-WM design study: arXiv 2512.24497 — Terver et al. (Meta FAIR, Inria). *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?*
- Ground-JEPA: Research Square rs-10511499. *Learning Physically Grounded Latent World Models for Zero-Shot Dynamics Generalization of Legged Robots.*
- SkyJEPA: arXiv 2606.23444. *Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors.*
- LeWM: arXiv 2603.19312 — Maes et al. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.*
- V-JEPA 2: Assran et al., 2025. DINO-WM: Zhou et al. PLDM: Sobal et al., 2025.
- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2022, pp. 91–100.
- Mittal et al. *Leveraging Symmetry in RL-based Legged Locomotion Control.* IROS 2024.
- Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* 2021.

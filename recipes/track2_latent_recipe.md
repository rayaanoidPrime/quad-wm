# Track 2 — Latent World Model for Quadruped Locomotion

**Base system:** DreamerV3 RSSM backbone, wired per WMP (World Model-based Perception)
**Stretch goal:** SWAP symmetry-equivariant world model
**Primary sensor:** depth images + proprioception
**Platform:** ANYmal D
**Simulator:** Isaac Lab

---

## 0. Why this base, and what the stretch goal changes

This is the highest-evidence track of the three.

| Component | Source | Status |
|---|---|---|
| DreamerV3 | Hafner et al., *Nature* 640(8059):647–653, 2025; arXiv 2301.04104 | Peer-reviewed, Nature. SOTA on 150+ tasks with a single fixed hyperparameter configuration. |
| WMP | Lai et al., ICRA 2025, pp. 11531–11537; arXiv 2409.16784 | Peer-reviewed ICRA. Real hardware (Unitree A1, Jetson NX). |
| DayDreamer | Wu et al., CoRL 2023, pp. 2226–2240 | Peer-reviewed. Dreamer methods on physical robots, complex locomotion with limited real-world interaction. |
| SWAP | arXiv 2606.19928 — Lan et al. | Preprint, June 2026. Real Apollo quadruped: 2.13 m gap, 1.63 m platform climb, reported as quadruped parkour records. No external replication. |
| DreamerV4 | arXiv 2509.24527 | Preprint. Minecraft-native. **No published legged locomotion result.** Not a contender for this track. |

**The distinction that drives the choice.** DreamerV3 + WMP is the right base for *robust terrain traversal*. SWAP is the right base for *extreme parkour*. SWAP's symmetry equivariance is not an optional add-on to the numbers — it is the reason the numbers are what they are. If parkour-class agility is not the target, the equivariance machinery is complexity you do not need. Keeping SWAP as a stretch goal is the correct sequencing.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $x_t$ | Observation: depth image + proprioception |
| $h_t$ | Deterministic recurrent state |
| $z_t$ | Stochastic latent state (discrete, categorical) |
| $\hat z_t$ | Prior sample (predicted without observing $x_t$) |
| $a_t$ | Action (target joint positions) |
| $r_t$ | Reward |
| $c_t$ | Continuation flag (episode not terminated) |
| $\phi$ | World model parameters |

---

## 2. Architecture (RSSM)

DreamerV3's RSSM encodes both stochastic latent states and deterministic history, enabling the model to represent environmental uncertainty and perform long-horizon latent rollouts under partial observability.

### 2.1 Components

$$
\begin{aligned}
\text{Recurrent model:} \quad & h_t = f_\phi(h_{t-1},\, z_{t-1},\, a_{t-1}) \\
\text{Encoder (posterior):} \quad & z_t \sim q_\phi(z_t \mid h_t,\, x_t) \\
\text{Dynamics (prior):} \quad & \hat z_t \sim p_\phi(\hat z_t \mid h_t) \\
\text{Decoder:} \quad & \hat x_t \sim p_\phi(\hat x_t \mid h_t,\, z_t) \\
\text{Reward head:} \quad & \hat r_t \sim p_\phi(\hat r_t \mid h_t,\, z_t) \\
\text{Continue head:} \quad & \hat c_t \sim p_\phi(\hat c_t \mid h_t,\, z_t)
\end{aligned}
$$

### 2.2 Observation wiring (WMP's contribution)

WMP is the cleanest application of RSSM to visual quadruped locomotion. Its key structural choice is replacing the two-stage teacher/student privileged-learning pipeline with a single jointly trained system:

- Proprioception + depth images → RSSM → latent state → policy → joint commands.
- World model and policy are trained end-to-end together in simulation.
- **No separate teacher policy is required.**

The rationale is that the student in classic distillation never recovers the full information the teacher had (privileged terrain geometry); the RSSM latent learns a compressed but informative representation that closes most of that gap without privileged access at test time.

### 2.3 Asynchronous update rates (WMP practical detail)

The world model update runs at a **lower frequency** than the action policy. This is a deliberate design decision, not an optimization. On WMP's Jetson NX deployment the depth-to-action pipeline latency was approximately 100 ms, accepted as a design constraint.

**Action for ANYmal D:** set the policy control rate to 50 Hz and the world-model update to a decimated rate (start at 10 Hz, i.e. every 5th control step). Record both rates explicitly in every experiment log — comparisons across runs with different decimation are not valid.

---

## 3. Training objectives

### 3.1 World model loss

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\left[\sum_{t=1}^{T} \Big( \mathcal{L}_{\text{pred}}^t + \beta_{\text{dyn}} \mathcal{L}_{\text{dyn}}^t + \beta_{\text{rep}} \mathcal{L}_{\text{rep}}^t \Big)\right]
$$

with

$$
\begin{aligned}
\mathcal{L}_{\text{pred}}^t &= -\ln p_\phi(x_t \mid h_t, z_t) \;-\; \ln p_\phi(r_t \mid h_t, z_t) \;-\; \ln p_\phi(c_t \mid h_t, z_t) \\[4pt]
\mathcal{L}_{\text{dyn}}^t &= \max\Big(1,\; \mathrm{KL}\big[\,\text{sg}(q_\phi(z_t \mid h_t, x_t)) \,\big\|\, p_\phi(\hat z_t \mid h_t)\,\big]\Big) \\[4pt]
\mathcal{L}_{\text{rep}}^t &= \max\Big(1,\; \mathrm{KL}\big[\,q_\phi(z_t \mid h_t, x_t) \,\big\|\, \text{sg}(p_\phi(\hat z_t \mid h_t))\,\big]\Big)
\end{aligned}
$$

**KL balancing.** The asymmetric weighting ($\beta_{\text{dyn}} \neq \beta_{\text{rep}}$) is what prevents posterior collapse. DreamerV3 defaults: $\beta_{\text{dyn}} = 0.5$, $\beta_{\text{rep}} = 0.1$. The $\max(1, \cdot)$ free-bits clipping prevents the KL from being driven to zero.

**Do not tune these.** DreamerV3's central empirical claim is that a single fixed hyperparameter configuration achieves SOTA across 150+ tasks. Use the published defaults as-is for the first full run. Only deviate after a baseline run has established that the defaults fail on your domain.

### 3.2 Robustness techniques (keep all three)

- **Symlog observation transform**, to stabilize training across input scales:
  $$\text{symlog}(x) = \text{sign}(x)\ln(|x| + 1), \qquad \text{symexp}(x) = \text{sign}(x)\big(\exp(|x|) - 1\big)$$
- **KL balancing**, as above.
- **Percentile return normalization**, so actor-critic updates are scale-invariant to task reward magnitude:
  $$S = \max\big(1,\; \text{Per}(R, 95) - \text{Per}(R, 5)\big), \qquad \tilde R = R / S$$

### 3.3 Actor-critic in imagination

Actor and critic train entirely inside the world model via imagined rollouts, which are substantially cheaper than environment steps. Gradients flow through the imagined trajectory for the actor.

- Imagination horizon: start at $L = 15$.
- Critic target: $\lambda$-return with $\lambda = 0.95$, discount $\gamma = 0.997$.
- Actor loss includes an entropy regularizer to prevent premature determinism.

---

## 4. Reward specification

Same base stack as Track 1, so that cross-track comparison is valid. Any divergence must be documented.

- **Linear velocity tracking:**
  $$r_{v_{xy}} = \exp\left(-\frac{\|v_{xy}^{\text{cmd}} - v_{xy}\|_2^2}{\sigma_v}\right), \quad \sigma_v = 0.25$$
- **Angular velocity tracking:**
  $$r_{\omega_z} = \exp\left(-\frac{(\omega_z^{\text{cmd}} - \omega_z)^2}{\sigma_\omega}\right), \quad \sigma_\omega = 0.25$$
- **Regularization:** torque $-\|\tau\|_2^2$, joint acceleration $-\|\ddot q\|_2^2$, action rate $-\|a_t - a_{t-1}\|_2^2$.
- **Terrain curriculum:** flat → slopes → stairs → gaps → climbs, advancing on a traversal threshold (Rudin et al., CoRL 2022).

### 4.1 AMP discriminator (optional, adds naturalness)

Adversarial Motion Priors (Peng et al., ACM TOG 40(4)) replace hand-engineered naturalness rewards with a learned discriminator distinguishing agent state transitions from a reference motion dataset.

$$
\mathcal{L}_D = \mathbb{E}_{d^{\mathcal{M}}}\big[(D(s,s') - 1)^2\big] \;+\; \mathbb{E}_{d^{\pi}}\big[(D(s,s') + 1)^2\big] \;+\; \frac{w_{\text{gp}}}{2}\,\mathbb{E}_{d^{\mathcal{M}}}\big[\|\nabla_s D(s,s')\|^2\big]
$$

$$
r^{\text{AMP}}_t = \max\Big(0,\; 1 - \tfrac{1}{4}\big(D(s_t, s_{t+1}) - 1\big)^2\Big)
$$

**Reference dataset.** GrandTour's real ANYmal D locomotion trajectories are a legitimate reference source for the discriminator — real traversal across varied terrain is arguably better grounded for a quadruped than human MoCap retargeted to a quadruped embodiment. Use state transitions $(s, s')$ extracted from the logged proprioceptive streams.

**Decision:** run the first full experiment **without** AMP. Add it in E2.5 as an explicit ablation. Adding a GAN objective to an already-complex training loop before the baseline is stable makes debugging intractable.

---

## 5. Data

| Source | Role |
|---|---|
| Isaac Lab rollouts | Entire RL/imagination loop. Only source with reward. |
| GrandTour (ANYmal D) | (a) AMP discriminator reference transitions, (b) real-world open-loop evaluation set |

Same mission-level split protocol as Track 1. Verify GrandTour's actual sensor suite and rates from the dataset documentation before building the loader — do not assume depth availability or synchronization.

---

## 6. Experiment plan

### E2.0 — Baseline reproduction
- Run DreamerV3 with published default hyperparameters on the Isaac Lab ANYmal D flat-terrain task, proprioception only (no depth).
- **Purpose:** establish that the implementation is correct before adding perception. If DreamerV3 defaults cannot learn flat-terrain locomotion, the bug is in the wiring, not the method.
- **Gate:** stable velocity tracking on flat terrain.

### E2.1 — Add depth (WMP wiring)
- Add the depth encoder branch. Single jointly trained system, no teacher policy.
- **Gate:** policy exceeds the proprioception-only baseline on rough terrain.

### E2.2 — Update-rate decimation sweep
- Sweep world-model update decimation $\in \{1, 2, 5, 10\}$ relative to the 50 Hz control rate.
- **Metrics:** policy success rate, wall-clock training time, and simulated deployment latency.
- **Why it matters:** WMP accepted ~100 ms depth-to-action latency as a design constraint on Jetson NX. ANYmal D's onboard compute differs; establish your own latency budget empirically rather than inheriting theirs.

### E2.3 — Teacher–student comparison
- Arm A: WMP-style single-stage joint training (default).
- Arm B: classical privileged teacher → depth student distillation (RMA lineage, Kumar et al. 2021).
- **Purpose:** verify WMP's central claim on ANYmal D. WMP argues the RSSM latent closes most of the privileged-information gap without test-time privileged access. This has been shown on Unitree A1, not on ANYmal D.

### E2.4 — Imagination horizon sweep
- Sweep $L \in \{8, 15, 25, 40\}$.
- **Metric:** policy success and sample efficiency. Longer horizons compound model error; this locates the trade-off point for locomotion specifically.

### E2.5 — AMP ablation
- Toggle the discriminator reward on/off, using GrandTour reference transitions.
- **Metrics:** gait naturalness (report torque smoothness and foot-contact regularity as proxies), plus task success. Naturalness gains that cost task success are a real trade-off to document, not a failure.

### E2.6 — Real-data open-loop evaluation (GrandTour)
- Freeze world model. Feed held-out mission observations and logged commands. Compare predicted state against logged state estimates.
- **Note the asymmetry with Track 1:** DreamerV3 has a decoder, so you can *additionally* report reconstruction error here. Do **not** use reconstruction error for cross-track comparison — Track 1's JEPA has no decoder by construction. Use probe-space metrics for anything cross-track (see shared evaluation protocol).

### E2.7 — Stretch: SWAP symmetry equivariance

Only after E2.0–E2.6 are complete.

SWAP's argument: purely data-driven latent world models must redundantly encode left–right symmetric interactions as independent patterns, which inflates the learning burden and restricts the latent space's efficiency for downstream policies. SWAP embeds symmetry directly into both the world model and the actor-critic networks.

Formally, for the reflection group $G = \{e, \sigma\} \cong \mathbb{Z}_2$, every network layer $f$ must satisfy

$$f\big(\rho_{\text{in}}(g)\, x\big) = \rho_{\text{out}}(g)\, f(x) \qquad \forall\, g \in G$$

where $\rho_{\text{in}}, \rho_{\text{out}}$ are the representations of $G$ acting on the input and output spaces. For ANYmal D this means the left–right leg permutation composed with the appropriate sign flips on lateral velocity, roll, and yaw components.

- **Implementation note:** this is a hard architectural constraint, implemented by weight sharing and structured parameterization. It is **not** the same as the mirroring data augmentation of Mittal et al. (IROS 2024), which is the weaker form. Implement from SWAP's formulation.
- **SWAP architecture:** low-frequency symmetric equivariant world model paired with a high-frequency policy — structurally consistent with the decimation scheme in E2.2.
- **Evaluation:** mirrored-terrain transfer is the decisive test. SWAP demonstrates that policies trained exclusively on unilateral asymmetric terrains transfer directly to mirrored environments without fine-tuning. Replicate this protocol exactly (see shared evaluation protocol, EV6).
- **Target results for reference:** 2.13 m gap leap, 1.63 m platform climb on the Apollo quadruped, with zero-shot transfer to unseen outdoor environments. ANYmal D is a different platform with different mass and leg geometry — do not expect these numbers to transfer directly, and do not report them as your target.

---

## 7. Known risks and mitigations

| Risk | Signature | Mitigation |
|---|---|---|
| Posterior collapse | KL term pinned at the free-bits floor; prior and posterior indistinguishable | KL balancing is already the mitigation; verify $\beta_{\text{dyn}} \neq \beta_{\text{rep}}$ is actually applied in code |
| Reward-head dominance | World model fits reward well, dynamics poorly | Monitor per-term losses separately, not just the total |
| Depth rendering throughput | Training wall-clock explodes | Establish FPS budget before planning experiments; decimate world-model updates (E2.2) |
| Imagination model exploitation | Policy finds high-reward trajectories that are physically impossible | Monitor the gap between imagined return and real environment return; a widening gap is the signature |
| Hyperparameter drift | Team starts tuning DreamerV3 defaults | Freeze defaults until E2.0 gate passes; log every deviation |

---

## 8. References

- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature 640(8059):647–653, 2025. arXiv 2301.04104.
- Lai, Cao, Xu, Wu, Lin, Kong, Yu, Zhang. *World Model-based Perception for Visual Legged Locomotion.* ICRA 2025, pp. 11531–11537. arXiv 2409.16784.
- Wu, Escontrela, Hafner, Abbeel, Goldberg. *DayDreamer: World Models for Physical Robot Learning.* CoRL 2023, pp. 2226–2240.
- Lan, Wang, Li, Jiang, Fu, Su, Wong, Jin, Wang. *SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour.* arXiv 2606.19928.
- Nahrendra, Yu, Myung. *DreamWaQ.* ICRA 2023, pp. 5078–5084. arXiv 2301.10602.
- Peng, Ma, Abbeel, Levine, Kanazawa. *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.* ACM TOG 40(4).
- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes.* CoRL 2022, pp. 91–100.
- Mittal et al. *Leveraging Symmetry in RL-based Legged Locomotion Control.* IROS 2024.
- Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* 2021.
- Hafner et al. *Training Agents Inside of Scalable World Models* (Dreamer 4). arXiv 2509.24527. — listed for completeness; not used.

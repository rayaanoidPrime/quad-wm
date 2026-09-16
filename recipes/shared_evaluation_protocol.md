# Shared Evaluation Protocol — Cross-Track Comparison

Applies to all three tracks. This file defines the only metrics valid for **cross-track** comparison. Track-specific metrics (reconstruction error, video quality) stay inside their own track's report.

---

## 0. The core problem this protocol solves

The three model classes have structurally incompatible native outputs:

| Track | Native output | Pixel metrics available? |
|---|---|---|
| 1 — JEPA | Latent vector only, **no decoder by construction** | No |
| 2 — DreamerV3/WMP | Latent + decoded observation | Yes |
| 3 — Generative | Physical state (QWM) or pixels (DreamDojo) | Depends on path |

Any metric requiring a decoder is unavailable for Track 1 and therefore invalid for cross-track use. **The unifier is probe space:** train a small head that maps each model's latent to physical state, and measure everything there.

This design converges with two independent lines of work. Ground-JEPA uses a physics-inspired probe mapping latent representations to interpretable physical states (position, velocity, orientation) to enable physically grounded prediction and interpretability. SkyJEPA combines a latent dynamics model with a physics-inspired prober mapping frozen latents to interpretable state for long-horizon prediction. PSG-JEPA uses the same structure but as a *training-time* objective rather than an evaluation tool, and evaluates latent identifiability via probing as its first of three evaluation levels.

---

## 1. Probe definition

### 1.1 Target state vector

Fixed across all tracks:

$$s_t = \big[\, p_t^{\text{base}},\; v_t^{\text{base}},\; \omega_t^{\text{base}},\; g_t^{\text{base}},\; q_t,\; \dot q_t,\; c_t \,\big]$$

| Component | Dim (ANYmal D) | Description |
|---|---|---|
| $p_t^{\text{base}}$ | 3 | Base position |
| $v_t^{\text{base}}$ | 3 | Base linear velocity, base frame |
| $\omega_t^{\text{base}}$ | 3 | Base angular velocity, base frame |
| $g_t^{\text{base}}$ | 3 | Projected gravity vector, base frame |
| $q_t$ | 12 | Joint positions |
| $\dot q_t$ | 12 | Joint velocities |
| $c_t$ | 4 | Binary foot contacts |

Total: 40 dimensions.

### 1.2 Probe architecture — fixed, identical across all tracks

- **Linear probe:** single affine layer $z \mapsto W z + b$.
- **MLP probe:** 2 hidden layers, width 256, GELU activation.

**Report both.** The linear–MLP gap measures how *linearly accessible* the information is, not just whether it is present. PSG-JEPA reports exactly this pairing (linear / MLP Pearson $r$ per cell).

### 1.3 Non-negotiable probe rules

Violating any of these invalidates the comparison:

1. **Identical architecture and parameter budget across all tracks.** Otherwise you measure probe capacity, not world-model quality.
2. **Identical optimizer, learning rate, and training budget for every probe.**
3. **World model frozen.** No gradients into the encoder during probe fitting.
4. **Probe trained on a held-out split**, and evaluated on a third split disjoint from both. Fit-set performance is meaningless here.
5. **Same probe fitting data for every model being compared.**

---

## 2. EV1 — Rollout error growth in probe space

The primary cross-track metric.

$$\epsilon_k = \frac{1}{N}\sum_{i=1}^{N} \big\| h_\psi\big(\hat z_{t+k}^{(i)}\big) - s_{t+k}^{(i)} \big\|_2$$

where $\hat z_{t+k}$ is obtained by rolling the world model open-loop for $k$ steps from $z_t$ under logged/ground-truth actions.

- **Horizons:** $k \in \{1, 5, 12, 25, 50\}$.
- **$k = 12$ rationale:** approximately one gait cycle. Compute the exact ANYmal D gait-cycle step count at your control rate and substitute it; do not copy 12 blindly.
- **Report the curve, not a scalar.** The *error accumulation rate* is the world-model quality signal. Autoregressive error accumulation over long horizons is the failure mode these models exist to address.
- **Normalize per state-component** by that component's standard deviation on the evaluation set, so joint angles and base position contribute comparably.
- **Also report per-component breakdown.** A model can be excellent on joint angles and useless on base velocity; the aggregate hides this.

---

## 3. EV2 — Action-conditioning fidelity

The highest-value eval in this protocol, and the one most rarely reported.

From identical initial state $z_t$, roll out under action sequence $A = a_{t:t+k-1}$ and a perturbed sequence $A' = A + \delta$. Define:

$$\text{ASR}(k) = \frac{\big\| h_\psi(\hat z_{t+k}^{A}) - h_\psi(\hat z_{t+k}^{A'}) \big\|_2}{\big\| s_{t+k}^{A} - s_{t+k}^{A'} \big\|_2}$$

where the denominator is the true divergence measured by replaying both action sequences in the simulator from the same state.

- **Ideal:** $\text{ASR}(k) \approx 1$ — the model's response to an action change matches reality's.
- **$\text{ASR} \ll 1$:** the model ignores actions. This is the characteristic failure of video-pretrained generative world models, which can produce convincing rollouts while barely responding to the control input.
- **$\text{ASR} \gg 1$:** the model over-reacts to actions; rollouts will be unstable under planning.
- **Perturbation design:** use several $\|\delta\|$ magnitudes (small, medium, large relative to the action range) and report ASR as a function of both $k$ and $\|\delta\|$.

**Gate for Track 3 video-generative path:** if ASR is low, the model is a video generator rather than a world model. Do not proceed to distillation.

---

## 4. EV3 — Downstream control at fixed post-training budget

This is where world-model quality cashes out. It requires post-training, as expected.

### 4.1 Fixed protocol

Held identical across all tracks:

- World model **frozen**.
- Identical policy head architecture and parameter count.
- Identical RL algorithm.
- Identical budget: same number of imagination steps **and** same number of environment steps.
- Identical shared terrain suite (§4.3).
- Minimum 3 seeds. Report mean and standard deviation.

### 4.2 Two variants — run both

| Variant | Method |
|---|---|
| **Planning** | CEM over action sequences in latent space |
| **Learned policy** | Actor-critic trained in imagination |

**Why both.** Terver et al. found that planning works best with sampling-based search such as CEM, because it handles the discontinuities arising from contact and friction where gradient planners get stuck — and contact discontinuity is exactly the locomotion regime. But the two model classes have different natural affordances: reporting only one variant will systematically flatter one track.

**CEM configuration** (following LeWM's published setup as a starting point): sample 300 candidate action sequences of horizon $H$, roll each autoregressively through the predictor, score by L2 distance to the goal latent, refit a Gaussian to the top 30 elites, repeat for 10–30 iterations, execute the first action, replan.

$$\mu^{(i+1)}, \Sigma^{(i+1)} \leftarrow \text{fit}\Big(\text{top-}30\big\{ A^{(j)} \sim \mathcal{N}(\mu^{(i)}, \Sigma^{(i)}) \big\}_{j=1}^{300}\Big)$$

### 4.3 Shared terrain suite

Fixed across all tracks and all seeds. Adapted from the standard curriculum (Rudin et al., CoRL 2022):

| Tier | Terrain | Metric |
|---|---|---|
| 1 | Flat | Velocity tracking error |
| 2 | Rough / noisy | Velocity tracking error, fall rate |
| 3 | Slopes (to 25°) | Traversal success rate |
| 4 | Stairs (up and down) | Traversal success rate |
| 5 | Gaps | Max gap crossed |
| 6 | Steps / climbs | Max height climbed |

---

## 5. EV4 — Dynamics-shift robustness

Identical perturbation grid across all tracks. Adopting Ground-JEPA's protocol gives you a direct comparison point against its reported figure.

| Perturbation | Levels |
|---|---|
| Base mass | $-30\%$, $-15\%$, nominal, $+15\%$, $+30\%$ |
| Ground friction | $0.4\times$, $1.0\times$, $1.5\times$, $2.5\times$ |
| Actuator latency | 0 ms, 10 ms, 25 ms, 50 ms |
| **Compound** | mass $\pm30\%$ **and** friction $2.5\times$ **and** latency 50 ms simultaneously |

**Reported metric:**

$$\text{Retention} = \frac{\text{Performance under perturbation}}{\text{Performance at nominal}} \times 100\%$$

Ground-JEPA reports retaining 78% of nominal performance zero-shot under the compound condition. Use this as a reference point, noting it was measured on DeepMind Control Suite quadruped-walk, not on ANYmal D in Isaac Lab — the numbers are not directly comparable, only the protocol is.

---

## 6. EV5 — Real-data open-loop evaluation (GrandTour)

The differentiator. Very little published world-model work evaluates prediction against real logged robot data at this scale.

- **Data:** held-out GrandTour missions, split at the **mission** level. Frame-level splits leak because consecutive frames are near-duplicates.
- **Procedure:** feed real logged observations and real logged commanded actions. Roll the model open-loop. Compare predicted state to logged state estimates.
- **Metric:** same $\epsilon_k$ curve as EV1, computed on real data.
- **Report the sim-to-real probe gap explicitly:**
  $$\Delta_{\text{s2r}}(k) = \epsilon_k^{\text{real}} - \epsilon_k^{\text{sim}}$$
  This quantity — how much worse the world model predicts on real data than in simulation, per horizon — is the most informative single number in this protocol and is essentially unreported in the literature.

**Caveat to state in any write-up.** GrandTour's logged state estimates are themselves estimates, not ground truth. Characterize their uncertainty (from the dataset documentation) and do not report $\epsilon_k^{\text{real}}$ below that noise floor as a meaningful difference between models.

---

## 7. EV6 — Mirrored-terrain transfer

Cheap to run, and the cleanest available generalization-vs-memorization test.

- **Protocol (from SWAP):** train policies exclusively on unilateral asymmetric terrains, then evaluate directly on mirrored environments **without fine-tuning**.
- **Metric:** success rate on mirrored terrain divided by success rate on the original terrain.
- **Expected outcome:** Track 2's SWAP arm should win this by construction, since equivariance is a hard architectural constraint. The interesting result is the size of the gap for the non-equivariant models, and whether Track 1's symmetry augmentation (E1.5) recovers any of it.

---

## 8. EV7 — Compute and latency parity

Mandatory. Without it, comparing a ~1.3M-parameter JEPA against a multi-billion-parameter generative model produces a meaningless ranking.

Report for every model:

| Quantity | Unit |
|---|---|
| Total parameters (inference-time) | M |
| Training-only parameters (e.g. PSG-JEPA grounding heads) | M |
| Training compute | GPU-hours, with GPU type |
| Open-loop rollout throughput | FPS |
| Single-step inference latency | ms |
| Deployment latency, observation → action | ms |

**Note for Track 1:** PSG-JEPA's grounding objectives are applied only during training, leaving the inference architecture and computational cost unchanged. Report inference-time and training-only parameters separately so this property is visible in the table.

---

## 9. Metrics explicitly excluded from cross-track comparison

| Metric | Why excluded |
|---|---|
| FVD, PSNR, SSIM, LPIPS | Measure pixel fidelity; anti-correlate with control usefulness. Video VAE reconstruction objectives prioritize pixel-level accuracy and preserve low-level appearance detail that is often unnecessary for planning or policy learning. Sanity check only, within Track 3. |
| Observation reconstruction error | Unavailable for Track 1 by construction (no decoder). |
| World-model training loss | Not comparable across different objectives (JEPA latent MSE vs ELBO vs generative likelihood). |
| Wall-clock training time alone | Confounded by implementation quality and hardware. Report alongside GPU-hours, never instead. |

---

## 10. Reporting template

One row per model per eval. Every cell must carry a seed count and a dispersion measure.

```
model:            <track>-<variant>-<ablation arm>
seeds:            n >= 3
EV1  eps_k:       [k=1, 5, 12, 25, 50]  mean ± std, per-component breakdown attached
EV2  ASR(k,|d|):  matrix, mean ± std
EV3  planning:    per-terrain-tier success, mean ± std
EV3  policy:      per-terrain-tier success, mean ± std
EV4  retention:   per-perturbation and compound, %
EV5  eps_k real:  [k=...], plus Delta_s2r(k)
EV6  mirror:      ratio
EV7  compute:     params / GPU-h / FPS / latency
```

---

## 11. Statistical discipline

- **Minimum 3 seeds**, 5 preferred, for anything reported as a result.
- **Report dispersion always.** Ground-JEPA's headline quadruped-walk comparison is 510 ± 68 versus 450 ± 234 at $p < 0.05$ — a claimed improvement whose baseline standard deviation is wide enough to overlap it substantially. Do not produce results with that shape and call them conclusive.
- **State the test used.** For seed-level comparisons with small $n$, a non-parametric test (Mann–Whitney U) is more defensible than a t-test.
- **Correct for multiple comparisons** when sweeping ablation arms. A sweep of 8 arms at $p < 0.05$ produces a false positive roughly 1 time in 3 by chance alone.

---

## 12. References

- Ground-JEPA: Research Square rs-10511499. Physics probe; H=12 horizon; compound dynamics-shift protocol; 78% retention figure.
- SkyJEPA: arXiv 2606.23444. Physics-inspired prober on frozen latents.
- PSG-JEPA: arXiv 2608.06799 — Yan et al. Three-level evaluation (probing / planning / policy); linear and MLP probe reporting.
- Terver et al.: arXiv 2512.24497 (Meta FAIR, Inria). CEM over gradient planners under contact discontinuity; encoder choice findings.
- LeWM: arXiv 2603.19312 — Maes et al. CEM configuration (300 candidates, 30 elites, 10–30 iterations).
- SWAP: arXiv 2606.19928 — Lan et al. Mirrored-terrain transfer protocol.
- Rudin, Hoeller, Reist, Hutter. CoRL 2022, pp. 91–100. Terrain curriculum.

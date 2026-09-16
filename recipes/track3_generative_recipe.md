# Track 3 — Generative World Model for Quadruped Locomotion

**Primary path:** QWM (morphology-conditioned generative dynamics model)
**Alternative path:** DreamDojo / Cosmos video-generative post-training
**Platform:** ANYmal D
**Simulator:** Isaac Lab

---

## 0. Disambiguation — decide this before writing any code

"Generative world model" means two structurally different things in the current literature. They need different architectures, different data, and different evaluations. Pick one explicitly.

| | **State-generative** | **Video-generative** |
|---|---|---|
| Output | Physical state / observation vector | Pixels |
| Representative | QWM (arXiv 2604.08780) | DreamDojo (arXiv 2602.06949), Cosmos |
| Data need | Robot trajectories, simulator-generated | Massive video corpus + small labeled robot set |
| Native eval | State prediction error, policy success | Video quality + action controllability |
| Compute | Lab-feasible | Pretraining is not lab-feasible; post-training from released weights is |
| Fit for locomotion | Direct | Indirect |

**Recommendation: state-generative (QWM).** Reasoning is in §1.

---

## 1. Why QWM over DreamDojo for locomotion

DreamDojo is a real and strong result, but it is a **manipulation** foundation model.

- Its pretraining corpus is 44k hours of egocentric human video spanning daily scenarios with diverse objects and skills — described as the largest video dataset to date for world model pretraining, with approximately 96× more skills and 2,000× more scenes than the most diverse public robot learning datasets.
- Its stated capability after post-training on small-scale target robot data is a strong understanding of physics and precise action controllability, evaluated on OOD benchmarks for open-world, contact-rich tasks.
- Its distillation pipeline reaches 10.81 FPS for autoregressive prediction, supporting interaction for more than a minute in real time.

**The problem for your use case.** The physics prior learned in stage 1 is hand–object contact from a head-mounted camera. Legged locomotion requires terrain–foot contact, ground reaction forces, and whole-body momentum. These are different physics. If you adopt DreamDojo, your stage-2 post-training set (GrandTour) carries essentially the entire locomotion burden while you pay for a 2B/14B backbone that learned to model people manipulating objects. The claim that this recipe "ports directly to locomotion" is not supported by anything in the paper.

**QWM's fit is direct.** QWM conditions a single generative dynamics model on a scale-invariant morphology specification, and extracts the policy in imagination. Its motivating observation is exactly your problem: a dynamics model trained on an ANYmal-D quadruped fails on a Unitree Go1 because it overfits to one robot's embodiment rather than capturing locomotion dynamics shared across robots, so even a small change in actuator dynamics or limb length forces retraining from scratch.

Critically for you, QWM's evaluation already includes your platform: it structures zero-shot generalization into **morphological interpolation** (holding out robots with high structural similarity to the training cohort, specifically the Unitree Go1 and **ANYmal-D**, at z-score Euclidean distance $d < 1.15$) and **morphological extrapolation** (holding out the Unitree B2 as a geometric outlier due to significantly higher mass and distinct stance geometry). On the held-out Go1 and ANYmal-D, with policy and world-model weights frozen and the agent conditioned solely on the target robot's morphology embedding $\mu$ derived from its USD, QWM achieves locomotion competitive with a specialist PPO baseline trained exclusively on that single robot.

**A warning QWM itself provides.** QWM's paper reports that other world-model baselines were excluded from the zero-shot analysis because they **failed to converge on the training set**. Take this seriously: this class of model is fragile at the training stage. Budget for convergence debugging.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $\mu$ | Morphology embedding, derived from the robot's USD description |
| $s_t$ | Physical state (base pose/velocity, joint positions, joint velocities, contacts) |
| $a_t$ | Action (target joint positions) |
| $\hat s_{t+1}$ | Generated next state |
| $\pi_\theta$ | Policy extracted in imagination |
| $d(\mu_i, \mu_j)$ | z-score Euclidean distance between two morphology specifications |

---

## 3. Architecture (QWM path)

### 3.1 Morphology specification

- Extract scale-invariant physical traits from each robot's USD: limb lengths, mass distribution, stance geometry, actuator limits, joint counts.
- Normalize to z-scores across the training cohort so that $d(\mu_i, \mu_j)$ is comparable.
- **This vector is the conditioning signal.** It must be computable for an unseen robot without any trajectory data from that robot — that is the entire point.

### 3.2 Generative dynamics model

$$\hat s_{t+1} \sim p_\phi\big(s_{t+1} \mid s_t,\, a_t,\, \mu\big)$$

Trained on multi-embodiment rollouts. The conditioning on $\mu$ is what allows one model to cover a family of robots rather than one.

### 3.3 Policy extraction in imagination

QWM's central architectural argument: given a morphology specification, you can either feed it to a model-free policy, or feed it to a learned dynamics model and extract the policy in imagination. QWM argues for the second route. The baseline for the first route is **PME-PPO** — a model-free policy conditioned on the same $\mu$ vector, with no learned dynamics model. QWM reports PME-PPO achieving substantially lower zero-shot performance on both held-out platforms.

**You must run PME-PPO as a baseline.** Without it you cannot distinguish whether generalization comes from $\mu$-conditioning alone or from the world model's learned latent dynamics. This is the single most important control in this track.

---

## 4. Training cohort design

QWM's method depends on a *family* of robots, not one. You need a multi-embodiment training set.

- **Training cohort:** assemble robot descriptions spanning the morphology space. Publicly available quadruped USD/URDF descriptions include Unitree A1, Go1, Go2, B1, B2, Aliengo, and ANYmal C. Verify licensing and availability for each before planning.
- **Held-out interpolation set:** ANYmal D (your platform), Unitree Go1.
- **Held-out extrapolation set:** one geometric outlier (high mass, distinct stance geometry).
- **Training data:** Isaac Lab rollouts across the training cohort. This is generated, not collected — the cohort size is limited by simulation throughput and by available robot descriptions, not by hardware access.

**Design decision to make explicitly.** Holding ANYmal D out of training gives you QWM's zero-shot claim on your own platform, which is the scientifically interesting result. Including ANYmal D in training gives you a better-performing policy. Run both arms (E3.3) and report the gap.

---

## 5. Experiment plan (QWM path)

### E3.0 — Morphology specification pipeline
- Build the USD → $\mu$ extraction for every robot in the cohort.
- Compute the full pairwise $d(\mu_i, \mu_j)$ matrix. Confirm ANYmal D falls within the interpolation regime ($d < 1.15$) relative to ANYmal C, and that the chosen outlier falls outside it.
- **Gate:** the distance matrix reproduces the expected interpolation/extrapolation structure.

### E3.1 — Single-embodiment convergence check
- Train the generative dynamics model on ANYmal D alone, no morphology conditioning.
- **Purpose:** establish that the dynamics model converges at all before adding multi-embodiment complexity. QWM reports that competing world models failed to converge on the training set; confirm yours does not.
- **Gate:** state prediction error plateaus at a usable level over a 50-step horizon.

### E3.2 — Multi-embodiment training with $\mu$ conditioning
- Train on the full cohort with ANYmal D **held out**.
- **Gate:** training-set locomotion performance across cohort robots comparable to per-robot specialists.

### E3.3 — Zero-shot transfer to ANYmal D (headline experiment)

| Arm | Description |
|---|---|
| A | QWM, ANYmal D held out, weights frozen, conditioned on $\mu_{\text{ANYmal D}}$ |
| B | PME-PPO, same $\mu$, model-free — **the essential control** |
| C | Specialist PPO trained only on ANYmal D — empirical upper bound |
| D | QWM with ANYmal D included in training — performance reference |

- **Metric:** episode length, velocity tracking error, success rate on the shared terrain suite.
- **Interpretation:** the A-vs-B gap isolates the world model's contribution. The A-vs-C gap measures how much zero-shot costs you. Report both.

### E3.4 — Extrapolation
- Repeat E3.3 with the geometric outlier as the held-out robot.
- **Expectation:** degraded performance relative to interpolation. QWM structures its evaluation around exactly this distinction, so a clean degradation curve is the expected result, not a failure.

### E3.5 — Real-data evaluation (GrandTour)
- Freeze the model. Open-loop rollout on held-out GrandTour missions using logged actions.
- Compare generated state trajectories against logged state estimates.
- **This is where your data is a genuine differentiator** — QWM is evaluated in simulation. A real-data evaluation of a morphology-conditioned generative dynamics model on a held-out embodiment is, as far as the current literature shows, unreported.

### E3.6 — Sim-to-real deployment (stretch)
- Deploy the imagination-extracted policy on ANYmal D hardware.
- **Precondition:** E3.3 arm A must be within a defensible margin of arm C in simulation. Do not deploy a policy that has not cleared its simulated baseline.

---

## 6. Alternative path — video-generative (only if visual rollouts are required)

Choose this **only** if you specifically need pixel-level rollouts for teleoperation, human inspection, or visual policy evaluation. Otherwise §5 is strictly better for locomotion.

### 6.1 Three-stage structure (DreamDojo)

- **Stage 1 — Pretraining.** Video prediction conditioned on continuous latent actions as unified proxy actions, trained without ground-truth action labels, on large-scale video.
  **Do not attempt this stage.** 44k hours is not a lab-scale corpus. Start from the released 2B or 14B checkpoints (available on Hugging Face under NVIDIA's open model license).
- **Stage 2 — Post-training.** Fine-tune on small-scale target robot data so the latent action space aligns to the robot's real action space. **GrandTour's onboard footage is exactly what this stage wants** — real, diverse-terrain, embodiment-matched video, which is normally the scarce resource in this recipe.
- **Stage 3 — Distillation.** Compress for real-time autoregressive inference (DreamDojo reports 10.81 FPS with improved context consistency). Only needed if you require live interactive use rather than offline evaluation.

### 6.2 Cosmos as an alternative base

If pursuing the video-generative path, evaluate NVIDIA Cosmos alongside DreamDojo. Cosmos is positioned as a platform — pretrained world foundation models, a post-training pipeline for specializing to a target embodiment, plus tokenizers and video curation tools. Its pretraining corpus contains substantially more egomotion and terrain content than hand–object video, which is a better prior match for locomotion.

### 6.3 Mandatory evaluation for this path

Video-generative models commonly produce high-quality rollouts while responding weakly to the action input. **Run the action-conditioning fidelity test (EV2 in the shared evaluation protocol) before investing in stage 3.** If action sensitivity is low, the model is a video generator, not a world model, and no amount of distillation fixes that.

**Do not report FVD/PSNR/SSIM as a cross-track metric.** These measure pixel fidelity and anti-correlate with control usefulness. Sanity check only, never comparison.

---

## 7. Known risks and mitigations

| Risk | Signature | Mitigation |
|---|---|---|
| Training non-convergence | Loss plateaus high; rollouts diverge quickly | E3.1 gate before multi-embodiment; QWM itself reports baselines failing here |
| $\mu$-conditioning is doing nothing | QWM ≈ PME-PPO in E3.3 | This is the result the PME-PPO control exists to detect. Report it honestly if it occurs. |
| Insufficient cohort diversity | Extrapolation fails completely | Verify the $d(\mu_i,\mu_j)$ spread in E3.0 before training |
| USD availability/licensing | Cohort smaller than planned | Resolve in E3.0; the method degrades gracefully with cohort size but the claim weakens |
| (Video path) weak action conditioning | High visual quality, low ASR | EV2 before stage 3 |

---

## 8. References

- QWM: arXiv 2604.08780 — Danesh et al. *Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion.* (v1 Apr 2026, v2 Aug 2026.) Also circulated as *Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning.*
- DreamDojo: arXiv 2602.06949 — Gao, Liang, Zheng, Malik, Ye, Yu, Tseng, Dong, Mo, Lin, Ma, Nah, Magne, Xiang, Xie, Zheng, Niu, Tan, Zentner, Kurian, Indupuru, Jannaty, Gu, Zhang, Malik, Abbeel, Liu, Zhu, Jang, Fan (NVIDIA). *DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos.* Checkpoints on Hugging Face (`nvidia/DreamDojo`).
- NVIDIA Cosmos — world foundation model platform, pretraining + post-training pipeline, tokenizers, video curation.
- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes.* CoRL 2022, pp. 91–100.

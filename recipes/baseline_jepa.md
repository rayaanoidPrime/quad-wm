# Track 1 — JEPA-WM Baseline for Quadruped Locomotion

---

- **Model:** Action-conditioned JEPA world model (JEPA-WM) on a frozen visual encoder, with proprioception
- **Encoder:** Frozen V-JEPA 2.1 ViT-B, spatial (local) tokens; the same frozen encoder produces the visual targets
- **Predictor:** 12-layer transformer, AdaLN + RoPE, frame-causal, operating on visual tokens that are each widened by a concatenated proprioceptive embedding (width $D+P$, §3)
- **Conditioning:** Visual+proprio tokens (proprio concatenated per-token, not a separate token) + actions
- **Training:** 6-step autoregressive rollout loss, weighted sum of the teacher-forced and rollout terms (§4), MSE in latent space
- **Planner:** CEM with an L2 latent-space cost, used at inference time only
- **Platform:** ANYmal D
- **Data:** GrandTour (real) and Isaac Lab rollouts (simulation)
- **RL / reward loss:** None

---

## 0. Evidence status of each component

Read this before committing compute.

| Component | Source | Status | Risk |
|---|---|---|---|
| JEPA-WM recipe (predictor architecture, rollout loss, context length, proprioception, CEM + L2) | arXiv 2512.24497 (Terver et al., Meta FAIR + Inria), v3 (18 May 2026), now TMLR-published | Code, data and checkpoints released (facebookresearch/jepa-wms). Validated on manipulation and navigation, not locomotion. **The specific combination used here — depth 12, $W=7$, $K=6$, proprioception — is a set of one-at-a-time ablation winners; the study never trained this exact combination together.** The released DROID config is DINOv3-L/16, depth-12 AdaLN+RoPE, 2-step rollout, $W=3$, no proprioception, $\alpha=0$. | Low as a method; Medium for transfer to locomotion; **treat the combined config as this project's hypothesis, not a replication** |
| V-JEPA 2.1 encoder | arXiv 2603.14482 | Meta release. Dense local features are its stated improvement. Not one of the encoders evaluated in the JEPA-WM study. In the study's own encoder comparison, DINO encoders (and DINOv3 over DINOv2) clearly beat V-JEPA/V-JEPA-2, attributed to finer object segmentation; the released real-data model itself uses DINOv3. | **Medium; DINOv3 is treated as a co-baseline (§8)** |
| GrandTour dataset | arXiv 2602.18164 | Public dataset; 49 missions, over 5 h. Preprint under review at IJRR. Camera, sync, and proprio-topic questions resolved from docs in §2.2; the action-definition gate and the exact license still need a hands-on check. | Low |
| PSG-JEPA grounding (ablation, §8) | arXiv 2608.06799 (Yan et al.), Aug 2026 preprint | A third-party summary reports LIBERO-Goal and real-robot results; no locomotion evaluation. The HKUST-GZ affiliation and a "navigation" scope could not be confirmed independently. In the paper, the grounding heads act on the shared latent of an end-to-end LeWM; our frozen-encoder version supervising predictor outputs is an adaptation, not a reproduction. | Medium |
| Gait-cycle horizon (ablation, §8) | Ground-JEPA, Research Square rs-10511499, posted 29 Jul 2026 | Explicitly not peer-reviewed, not on arXiv. ~1.3M-parameter model. Sim-only; task names resemble DeepMind Control though the abstract doesn't name DMC. Headline result: quadruped-walk 510±68 vs. a 450±234 baseline — wide variance either way. | High |
| `jepa-wms` code license | Repo README | **CC BY-NC 4.0 — non-commercial.** `vjepa2` is MIT with Apache-2.0 on three files. No separate license was found for the V-JEPA 2.1 weights themselves. | Flag before any release (§7) |

**Interpretation.** The baseline deliberately combines the JEPA-WM study's ablation-winning components for real-world manipulation — 12-layer AdaLN+RoPE predictor, $W=7$ training context (the value the paper's own 6-step models used, not the generic default), 6-step rollout, proprioception via concatenation, CEM+L2 planning — into a single configuration the study itself never trained end-to-end. That combination is this project's explicit hypothesis, not a replication of any single released config. The released DROID checkpoint is meaningfully different (no proprioception, $W=3$, 2-step rollout, $\alpha=0$) and is not what this baseline reproduces. V-JEPA 2.1 is the deliberate encoder departure here, run alongside a DINOv3 co-baseline (§8), since both the paper's own comparison and its released model favor DINOv3.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $o_t$ | Observation at tick $t$: an RGB frame $I_t$ and proprioceptive state $s_t^{\text{prop}}$ |
| $s_t^{\text{prop}}$ | Proprioceptive state: base linear/angular velocity, gravity vector in base frame, joint positions $q_t \in \mathbb{R}^{12}$, joint velocities $\dot q_t$ |
| $E^{\text{vis}}_\phi$ | Frozen visual encoder (V-JEPA 2.1 ViT-B). `jepa-wms` doesn't support V-JEPA 2.1 natively (its `enc_type` is `dino` or `vjepa`), so this requires a thin wrapper — see §3.1, §9. |
| $E^{\text{prop}}_\theta$ | Shallow proprioceptive encoder, trained jointly with the predictor. Outputs a $P$-dimensional embedding (reference configs use $P$ in the 10–20 range; this spec uses $P=16$ as a representative value — confirm against the exact config being matched before implementation) |
| $A_\theta$ | Action encoder, trained jointly with the predictor |
| $E_{\phi,\theta}$ | Global state encoder $(E^{\text{vis}}_\phi, E^{\text{prop}}_\theta)$; $z_t = E_{\phi,\theta}(o_t)$ consists of $N=576$ visual tokens, **each of width $D+P$**: the proprioceptive embedding is broadcast-concatenated onto every visual token, widening it from $D=768$ to $D+P$ (e.g. $784$ with $P=16$). This matches the reference's `proprio_encoding: feature` mode. There is **no** separate proprioceptive token. |
| $P_\theta$ | Predictor |
| $\Delta t$ | World-model tick; all streams are resampled to it (start with $0.2$ s, i.e. 5 Hz) |
| $f$ | Control steps per tick (frameskip) |
| $a_t \in \mathbb{R}^{12 f}$ | Action over tick $t \to t+1$: the $f$ commanded joint-position targets within the tick, concatenated |
| $W$ | Maximum training context, $\mathbf{W = 7}$. Matches the value the paper's own 6-step models used ($K=6$ models trained with a larger context than the generic default) — see §4.3. |
| $W^t$ | Rollout context cap *during training* (`ctxt_window_train_rollout` in the reference), $\mathbf{W^t = 3}$. Distinct from $W$: even though up to $W$ ground-truth ticks are available, the context fed back into the predictor during the autoregressive rollout is capped at $W^t$. |
| $K$ | Number of training rollout steps, $K = 6$ |
| $W^p$ | Maximum context at planning time, $W^p = 2$, with $W^p \le W$ |
| $H$ | Planning horizon (ticks) |
| $\alpha$ | Weight of the proprioceptive term in the planning cost |
| $L$ | Latent-space dissimilarity: MSE in training, L2 distance in planning |

---

## 2. Data

### 2.1 Sources

| Source | Role | Notes |
|---|---|---|
| GrandTour (ANYmal D) | (a) Predictor training corpus (Stage A), (b) real-world held-out evaluation | Front CoreResearch camera (`alphasense_front_center`, 1440×1080, 10 fps, PTP-synced) plus proprioception (`anymal_state_actuator`, `anymal_state_state_estimator`, 400 Hz reference rate). No reward and no privileged terrain labels. Cameras mount on the Boxi payload, not the ANYmal base. |
| Isaac Lab rollouts | Mixed training (Stage B); privileged-label probes; simulated planning evaluation | Rollouts from a fixed pretrained ANYmal D locomotion controller trained outside this track. Randomize velocity commands, add action noise and random pushes, randomize terrain, friction, mass, lighting and textures. Log RGB, proprioception, joint-position targets, terrain height, friction and contacts. Privileged labels are used only for probes, never in a training loss. |

### 2.2 What the dataset documentation resolves — and what it doesn't

**Resolved from GrandTour's docs and recorder repo:**

- **Camera:** use the front **CoreResearch** camera — global shutter, 1440×1080, 10 fps, ~126°×92° FoV, PTP-synced, topic `alphasense_front_center`. At a 5 Hz tick this is every second frame. (An HDR alternative exists — TierIV HDR, 1920×1280, 30 fps, rolling shutter, hardware-triggered, topic `hdr_front` — for later if a higher native rate is needed. The ZED2i is excluded: 15 fps with unreliable timestamps.)
- **Mounting and lenses:** cameras are on the Boxi payload, not the ANYmal base — use the Boxi→ANYmal extrinsics from CAD, not the base frame directly. CoreResearch and HDR both use a fisheye-type distortion model, so matching the Isaac Lab camera needs rectification (or a matching distortion model) plus FoV, and a 4:3→square crop decision (V-JEPA 2.1 wants square 384×384 input).
- **Sync:** CoreResearch is PTP-synced. ANYmal's own sensors sync internally and reach Boxi via NTP, so joint-to-camera alignment is millisecond-level — fine at 5 Hz, but measure it rather than assume it.
- **Proprio logging:** `anymal_state_actuator` and `anymal_state_state_estimator`, with a 400 Hz reference rate for both (from a recorder health-check config, not dataset documentation directly — indicative, not guaranteed).

**Still open — needs a hands-on check, not settled by documentation (this needs live access to Hugging Face to open a Zarr and check directly):**

- **Action gate.** GrandTour's paper text only mentions commanded *velocity*, logged separately as `anymal_command_twist` (the operator's twist command) — that is not what this recipe uses as the action. `anymal_state_actuator` is a `SeActuatorReadings` array; each of the 12 readings has a `state` block and a `commanded` block (mode, position, velocity, joint_torque, PID gains), and the HF converter writes both to Zarr (`NN_command_position`, `NN_command_mode`, …). So joint-position targets are *probably* present, at roughly 400 Hz rather than policy rate — but this is unconfirmed. **Before training:** on one mission, check that `NN_command_mode` is a joint-position mode and that `NN_command_position` actually varies, then measure the real update rate before fixing $f$. The $f=10$ / 50 Hz-controller assumption in §2.3 is a placeholder, not a verified fact.
- **License.** The HF card says MIT; the project site says the data is CC BY-SA 4.0 and the software is MIT. Resolve which applies before releasing any derived data (§7).
- Also not verified in this pass: GrandTour's Zarr contents generally, and Appendices E–G of the JEPA-WM paper.

### 2.3 Tick and action definition

Resample all streams to the world-model tick $\Delta t$ (start with 5 Hz). The action is the commanded joint-position targets, stacked over the tick (frameskip stacking, as in DINO-WM): $a_t \in \mathbb{R}^{12 f}$, where $f$ is the number of control steps per tick (placeholder: $f = 10$ for an assumed 50 Hz controller at a 5 Hz tick — **unverified, see §2.2**). The action is not reduced to a lower-dimensional command, and it has the same definition in GrandTour and Isaac Lab. Standardize actions and proprioceptive states per dimension with training-set statistics.

### 2.4 Split protocol

Partition GrandTour by *mission*, not by frame. Consecutive frames within a mission are near-duplicates, so frame-level splits leak. Reserve at least 20% of missions (at least 10 of the 49), spanning all terrain and illumination conditions present, as a held-out set that is never touched during training, probe fitting, or hyperparameter selection. In simulation, hold out terrain seeds and terrain types.

### 2.5 Feature caching

The encoder is frozen, so visual tokens are computed once per tick and cached; no image augmentation is applied in the baseline. Cache size is (frames) × $N$ × 768 × 2 bytes in fp16. With 576 tokens per frame, 5 Hz over 5 h is about 80 GB per camera (about 160 GB at 10 Hz). This is independent of the training context length $W$ — the cache stores one entry per frame and is sliced at use time. Since `jepa-wms` has no native V-JEPA 2.1 path, the wrapper (§3.1) needs to sit in front of this caching step.

---

## 3. Architecture

### 3.1 Visual encoder (frozen)

- **Model:** V-JEPA 2.1 ViT-B/16 (80M parameters, distilled from ViT-G), released checkpoint `vjepa2_1_vitb_dist_vitG_384.pt` (PyTorch Hub entry `vjepa2_1_vit_base_384`), weights frozen. Input resolution is the checkpoint's native $384 \times 384$.
- **Tokens:** Each frame is encoded on its own with the encoder's native image tokenizer: feed a 5-D tensor $(B,3,1,384,384)$; the image path uses a per-frame patch embed with an image-modality embedding. The default forward pass returns the last block's LayerNorm'd tokens, $(B,576,768)$ for ViT-B — confirmed against the checkpoint. The latent is not pooled to a single vector.
- **Wrapper needed:** `jepa-wms`'s `enc_type` is `dino` or `vjepa` only, with no native V-JEPA 2.1 path, so this recipe requires a thin wrapper exposing the call above.
- **Targets:** The targets are embeddings of future frames from the same frozen encoder. There is no EMA target encoder, no stop-gradient and no anti-collapse regularizer needed here, because the visual latent cannot change during training.
- **Note on duplicated frames.** The JEPA-WM study encoded each frame for its video encoders (V-JEPA, V-JEPA-2) by duplicating it and encoding the pair as a two-frame video. That's moot here: V-JEPA 2.1 has a native image tokenizer, which this recipe uses instead.

### 3.2 Proprioceptive and action encoders

$E^{\text{prop}}_\theta$ embeds $s_t^{\text{prop}}$ into a $P$-dimensional embedding ($P=16$, within the reference's reported 10–20 range). This embedding is **broadcast-concatenated onto every one of the $N$ visual tokens**, widening each token from $D=768$ to $D+P$ — matching the reference's `proprio_encoding: feature` mode. There is no separate proprioceptive token in the sequence. $A_\theta$ embeds $a_t$ into the action embedding used for conditioning; both encoders are shallow (linear or small MLP) and trained jointly with the predictor.

### 3.3 Predictor

- **Architecture:** 12-layer transformer, hidden width $D+P$ (e.g. $784$), AdaLN conditioning and RoPE positional embeddings. Pick $P$ so $D+P$ divides evenly by the chosen head count (e.g. 16 heads × 49 = 784).
- **Inputs per timestep:** the $N$ visual tokens, each already widened with the proprioceptive embedding (no separate proprio token); actions condition every block through AdaLN.
- **Frame-causal attention mask.** The predictor is trained simultaneously to predict from every context length $w = 0, \dots, W-1$, and predictions for all timesteps in a sequence are produced in parallel during teacher-forced training.
- **Outputs:** predicted next-step tokens of width $D+P$ each. The first $D$ dimensions of each token are the predicted visual features; the last $P$ dimensions are the predicted proprioceptive embedding — the same target value is broadcast across all $N$ tokens (see §4.1 for how this becomes a loss).

The predictor, $E^{\text{prop}}_\theta$ and $A_\theta$ are the only trainable parameters.

---

## 4. Training objective

There is no RL, reward, value or policy loss anywhere in training.

### 4.1 One-step (teacher-forced) loss

$$\mathcal{L}_1 = \frac{1}{B}\sum_{b=1}^{B} \Big[ L_{\text{vis}} + L_{\text{prop}} \Big]$$

where:
- $L_{\text{vis}}$ is the MSE between the predicted and target $D$-dimensional visual slice, averaged per-token over the $N$ tokens and per-dimension — target is $E^{\text{vis}}_\phi(o^b_{t+1})$, frozen by construction.
- $L_{\text{prop}}$ is the MSE between the predicted and target $P$-dimensional proprio slice, averaged the same way over the $N$ (broadcast) tokens — target is $\text{sg}\big[E^{\text{prop}}_\theta(s^{b,\text{prop}}_{t+1})\big]$.

Visual and proprio terms are summed with **equal weight**; there is no separate proprio weight in training (that's specific to planning, §6.1).

**Stop-gradient decision (departs from the reference).** In the reference, only rollout-step ($k\ge2$) targets are detached; the teacher-forced proprio target is not, which is a genuine collapse risk since prediction and target both flow through the same trainable $E^{\text{prop}}_\theta$. This spec adds a stop-gradient ($\text{sg}[\cdot]$ above) to the teacher-forced proprio target as well, removing that trivial solution. This is a deliberate departure from the reference, not a replication of it — monitor via the E1.1 variance check (§6, §7).

### 4.2 $k$-step rollout losses

For $k \ge 1$, unroll the predictor autoregressively for $k$ steps from the context $o_{t-W+1:t}$:

$$\hat z_{i+1} = P_\theta\big(\hat z_{i-w:i},\, A_\theta(a_{i-w:i})\big), \quad i = t, \dots, t+k-1$$

The context slides: it holds ground-truth embeddings at first and the model's own predictions once they are available. **During the rollout, the context fed into the predictor is capped at $W^t=3$ ticks**, even though up to $W=7$ ground-truth ticks exist for the initial context — this cap (the reference's `ctxt_window_train_rollout`) is separate from the training context $W$. The $k$-step loss compares the $k$-th prediction with the encoded future frame:

$$\mathcal{L}_k = \frac{1}{B}\sum_{b=1}^{B} \Big[ L_{\text{vis}} + L_{\text{prop}} \Big]\Big(P_\theta\big(\hat z^b,\, A_\theta(a^b)\big),\; \text{sg}\big[E_{\phi,\theta}(o^b_{t+k})\big]\Big)$$

with all $k\ge1$ rollout targets detached (both visual and proprio), and visual/proprio weighted equally as in §4.1. Each $\mathcal{L}_k$ uses truncated backpropagation through time: the accumulated gradient through earlier rollout steps is discarded, and only the error of the last prediction is backpropagated. $\mathcal{L}_1$ is the teacher-forced loss of §4.1.

### 4.3 Total

$$\mathcal{L} = \frac{1}{K+1}\,\mathcal{L}_1 \;+\; \frac{1}{K}\sum_{k=2}^{K}\mathcal{L}_k$$

With $K=6$: $\mathcal{L} = \frac{1}{7}\mathcal{L}_1 + \frac{1}{6}\big(\mathcal{L}_2+\mathcal{L}_3+\mathcal{L}_4+\mathcal{L}_5+\mathcal{L}_6\big)$ — this replaces a plain sum with the reference's actual weighting. Apply gradient-norm clipping at 1 during optimization (matches the reference).

**Sample span (decision).** With $W=7$ and $K=6$, each training sample spans $W+K=13$ consecutive ticks: context ticks $t-6,\dots,t$ and target ticks $t+1,\dots,t+6$. Note this is a practical approximation, not a literal replication: the reference frames a training slice as $W{+}1$ ticks with rollouts starting from a random prefix inside it (Appendix D), and how it chains slices together to cover $K=6$ steps of rollout couldn't be fully reconstructed from documentation alone. The straightforward "context + $K$ rollout targets" scheme above is what this baseline actually implements.

---

## 5. Training procedure

- **Stage A — GrandTour.** Train the predictor, $E^{\text{prop}}_\theta$ and $A_\theta$ on train-split GrandTour missions with $\mathcal{L}$ (§4.3), grad-norm clipped at 1.
- **Stage B — Mixed.** Continue on batches mixed from GrandTour and Isaac Lab (start 50/50), so simulated RGB is in-distribution for the simulated evaluations without discarding the real-data prior.
- **Freeze** the trained world model after Stage B. Every evaluation in §6 uses the frozen model.

---

## 6. Planning and evaluation

### 6.1 Planner (inference time only)

CEM is used only at inference time, on logged or simulated data for evaluation. It is not part of training and is not run on the robot.

- **Objective:** given the current observation $o_t$ and a goal observation $o_g$, find the action sequence $a_{t:t+H-1} \in \mathbb{R}^{H \times 12f}$ that minimizes

$$L^p_\alpha(o_t, a_{t:t+H-1}, o_g) = \big(L_{\text{vis}} + \alpha L_{\text{prop}}\big)\big(F_{\phi,\theta}(o_t, a_{t:t+H-1}),\; E_{\phi,\theta}(o_g)\big)$$

where $F_{\phi,\theta}$ unrolls the predictor for $H$ steps from $o_t$ and $L$ is the L2 distance in embedding space.
- **Proprioceptive weight:** $\alpha = 0.1$, the value the JEPA-WM study uses for models trained with proprioception (it sets $\alpha=0$ only on DROID/Robocasa, to match a vision-only baseline — not applicable here).
- **Context:** the planner rolls out with a sliding context of $W^p = 2$ frames, satisfying $W^p \le W$.
- **CEM hyperparameters (from the reference's own eval configs, not generic defaults):** population 300, 10 elites, 15 iterations (as used for DROID/Metaworld; 30 for Push-T), zero-mean initialization with std = `var_scale` (1.0 generally; 0.1 for the unnormalized DROID case), norm-clipped samples, no momentum term, cost on the final unrolled state only. Use planning horizon $H=6$ ticks (matching this baseline's $K$) with 6 executed steps, unless E1.3 results say otherwise. The largest action-space size validated in the reference is roughly 60–120 dimensions; at $f=10$ this recipe's search space is $H\times12f = 6\times120=720$ dimensions, well past anything the reference actually tested — scale population/iterations up from the numbers above and treat this as an open tuning problem, not a solved default.
- **Before scaling planning experiments, confirm the licensing gate (§7):** `jepa-wms` is CC BY-NC 4.0.

### E1.0 — Infrastructure and data validation (no science)
- Confirm GrandTour RGB and proprioception load and are time-aligned; settle the still-open items in §2.2 (action gate, license).
- **Gate on the action definition:** confirm GrandTour logs joint-position commands at a rate compatible with $f$ (§2.2). If it does not, resolve this before training rather than substituting a different action.
- **Gate on licensing:** confirm `jepa-wms`'s CC BY-NC 4.0 code license and the GrandTour MIT/CC BY-SA 4.0 conflict are compatible with the intended use of this work before investing further compute.
- Confirm Isaac Lab ANYmal D renders RGB and that the fixed controller produces stable rollouts. Measure simulation throughput (FPS) with camera rendering on.
- Extract and cache V-JEPA 2.1 tokens via the wrapper (§3.1); record $N$ and cache size.
- **Gate:** aligned data, defined action, cached features, known FPS, licensing resolved.

### E1.1 — Predictor sanity and baselines
- Compare the trained predictor against a persistence baseline, $\hat z_{t+k} = z_t$, at every rollout step $k = 1, \dots, 6$. The predictor must beat it.
- Log embedding-space error at each rollout step on train and held-out missions.
- Monitor the variance of the proprioceptive slice across the batch (the collapse risk in §7 — this is exactly what the added stop-gradient in §4.1 is meant to prevent).
- **Gate:** the predictor beats persistence at all six steps on held-out missions.

### E1.2 — Open-loop real-data evaluation
- On held-out GrandTour missions, roll the frozen world model out open-loop from logged RGB, proprioception and logged actions.
- Report embedding error and proprioceptive decoding error by rollout step, by terrain type, and against the persistence baseline. The proprioceptive decoder is a small head fit on train-split missions only.
- Fit probes on frozen tokens (train-split missions only) for physical state, reported per component: gravity direction, base velocity, joint positions and velocities. In simulation, also probe the privileged labels (terrain height, friction, contacts).
- **This is the differentiator.** Very few world-model papers evaluate prediction against real logged data at this scale. Report it regardless of whether it flatters the model.

### E1.3 — Planning evaluation (offline CEM)
- **Action recovery on GrandTour:** for held-out missions, take $o_t$ and a goal frame $o_g = o_{t+H}$, run CEM, and compare the planned actions with the logged actions (L1 error, rescaled into a score to maximize).
- **Simulated goal reaching:** goal frames drawn from held-out Isaac Lab trajectories; the success criterion is defined once E1.0 is complete.
- Report results with $\alpha = 0.1$ and against the persistence baseline where applicable.

### E1.4 — Downstream RL evaluation (deferred)
The RL evaluation on the frozen world model is part of the shared evaluation suite. Its specification is deferred and will be added here.

---

## 7. Known risks and mitigations

| Risk | Signature | Mitigation |
|---|---|---|
| Untested hyperparameter combination | Results may not match the paper's headline numbers even with a faithful implementation, since depth-12 + $W=7$ + $K=6$ + proprioception was never trained together as one config in the reference, and the released DROID model uses none of these (no proprio, $W=3$, 2-step, $\alpha=0$) | Treat as this project's core hypothesis, not a replication; gate on E1.1–E1.3 before scaling; DINOv3 co-baseline (§8) helps isolate the encoder's contribution |
| Encoder domain gap | Probe $R^2$ low on GrandTour: outdoor egocentric legged video has motion blur and vibration V-JEPA 2.1 wasn't trained for | E1.2 probes early; DINOv3 co-baseline (§8), promoted from a secondary reference arm |
| V-JEPA 2.1 untested in JEPA-WM | Predictor underperforms persistence despite good encoder probes | E1.1 gate; DINOv3 co-baseline (§8) |
| Hyperparameters tuned on manipulation | Large train/held-out mission gap with only about 5 h of real data | Monitor the gap; sweeps in §8 |
| Compounding rollout error | Embedding error grows quickly across steps 1–6 | Report error by step; the multi-step loss is the mitigation; the trade-off in $K$ is a sweep (§8) |
| Proprioceptive-target collapse | The proprioceptive slice's variance goes to zero, because in the reference its teacher-forced target comes from the same trainable encoder as the prediction | **Stop-gradient added to the teacher-forced ($k=1$) proprio target, in addition to the rollout targets the reference already detaches (§4.1) — a deliberate departure from the reference.** Monitor variance in E1.1 regardless. |
| High-dimensional action search | CEM over $H \times 12f$ dimensions converges poorly; the reference's largest validated search space is ~60–120 dims vs. this recipe's ~720 | Budget population and iterations accordingly (§6.1); report action recovery before any other planning metric |
| Sim-to-real appearance gap | Good simulated metrics, poor GrandTour metrics | Stage B mixing; lighting and texture randomization; camera matching (§2.2) |
| Action or timing mismatch | Poor prediction even at $k=1$ | E1.0 gate on the action definition; verify synchronization and resampling |
| Mission leakage | Held-out results inflated | Mission-level split (§2.4) |
| Cache size and predictor cost | 576 tokens × 7 context frames ≈ 4,032 visual tokens for the context window alone per training sample; a full 13-tick sample (context + 6 rollout targets) touches on the order of 7,500 visual tokens end-to-end; predictor width is also now $D+P$ rather than $D$ | The pooled-token variant in §8 (144 tokens per frame) |
| **License conflict** | `jepa-wms` is **CC BY-NC 4.0** (non-commercial); GrandTour's HF card (MIT) and project site (data CC BY-SA 4.0, software MIT) disagree with each other | Confirm intended use is non-commercial, or plan a from-scratch reimplementation of the JEPA-WM recipe before any commercial release; resolve the GrandTour MIT vs. CC BY-SA 4.0 conflict before releasing any derived data (E1.0 gate) |
| Preprint drift | The JEPA-WM paper or code revises its method or defaults | Re-check the latest arXiv version and the released code before implementing from paper text |

---

## 8. Ablations and sweeps (after the baseline; not a priority)

**Ablations**
- **Reference encoder — run alongside the baseline, not after it.** Frozen DINOv3 (and DINOv2) versus V-JEPA 2.1, using the same cached-feature pipeline. Given the reference paper's own encoder comparison favors DINOv3 and the released real-data model uses DINOv3 rather than V-JEPA 2.1, this is a co-baseline (§0), not a secondary sweep.
- **PSG grounding heads** (PSG-JEPA, arXiv 2608.06799). A state head grounds latents in proprioceptive state and a transition head grounds latent pairs in multi-horizon joint-angle change. In the source paper these heads act on the shared latent of an end-to-end LeWM; with our frozen encoder they instead supervise the predictor's outputs — this is an adaptation of PSG-JEPA, not a reproduction of it, and its LIBERO-Goal / real-robot results don't include a locomotion evaluation to compare against.
- **Joint encoder + predictor training.** Unfreeze the encoder. This requires an anti-collapse mechanism, for example SIGReg as in LeWM (arXiv 2603.19312), or EMA with a variance/covariance regularizer.
- **Multi-horizon prediction and gait-cycle horizon.** Multi-horizon targets, with a horizon set covering one gait cycle, $H_{\text{gait}} = \operatorname{round}(T_{\text{gait}} / \Delta t)$, where $T_{\text{gait}}$ is measured on the simulation controller and checked against GrandTour logs. Ground-JEPA's reported gain (quadruped-walk 510±68 vs. 450±234) is sim-only, not peer-reviewed, and has wide variance — treat it as a hypothesis, with at least 3 seeds.

**Hyperparameter sweeps**
- Predictor depth $\{6, 12\}$ × rollout steps $K \in \{2, 6\}$ × training context $W \in \{3, 7\}$.
- Tick rate $\{5, 10\}$ Hz.
- Reference encoder: frozen DINOv3 (and DINOv2) versus V-JEPA 2.1, using the same cached-feature pipeline (see co-baseline note above).
- Token grid: full grid (576 tokens per frame) versus a $2 \times 2$-pooled grid (144 tokens per frame).
- Planning: CEM population and iterations, planning horizon $H$, and planning context $W^p$.
- Stage B mixing ratio between GrandTour and Isaac Lab.

Held fixed, not swept: $\alpha = 0.1$, the action definition, the duplicated-frame question (§3.1, moot for V-JEPA 2.1), and the proprio concat-and-widen encoding (§3.2).

---

## 9. Matching the reference implementation

- **Proprio encoding and loss weighting:** concatenated feature, widening $D\to D+P$ (§3.2–§3.3); equal-weighted visual/proprio MSE inside a total loss weighted $\mathcal{L}_1/(K{+}1) + \sum_{k\ge2}\mathcal{L}_k/K$, grad clip 1 (§4).
- **Training slice, prefix, and rollout context cap:** treated as a practical approximation, not a literal replication — $W=7$, $W^t=3$ (§1, §4.2–§4.3). The reference's own slice-chaining mechanism (Appendix D) couldn't be fully reconstructed from documentation alone.
- **Planning defaults:** population 300, 10 elites, 15 (or 30 for Push-T) iterations, `var_scale`-based init, no momentum, cost on final state only (§6.1).
- **V-JEPA 2.1 tokenizer:** confirmed. 5-D input tensor $(B,3,1,384,384)$; per-frame patch embed with an image-modality embedding; default forward returns $(B,576,768)$ last-block LayerNorm'd tokens for ViT-B; checkpoint file, hub name, and 80M param count all match this spec. `jepa-wms` needs a wrapper since it has no native `enc_type` for V-JEPA 2.1 (§3.1).
- **Duplicated-frame question:** moot — V-JEPA 2.1 has a native image path (§3.1).
- **Licenses:** `jepa-wms` is CC BY-NC 4.0 (flagged as a risk, §7); `vjepa2` is MIT with Apache-2.0 on three files; no separate license was found for the V-JEPA 2.1 weights in the README — re-check before any release.

---

## 10. References

- JEPA-WM design study: arXiv 2512.24497 — Terver, Yang, Ponce, Bardes, LeCun. *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?* v3 (18 May 2026), TMLR-published. Code, data and weights: github.com/facebookresearch/jepa-wms (CC BY-NC 4.0).
- V-JEPA 2.1: arXiv 2603.14482. *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.* Code: github.com/facebookresearch/vjepa2 (MIT, three files Apache-2.0).
- V-JEPA 2: Assran et al., 2025. DINO-WM: Zhou et al., 2024.
- GrandTour: arXiv 2602.18164. *GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation.* Under review at IJRR; license listed as MIT on the HF card and CC BY-SA 4.0 (data) / MIT (software) on the project site — unresolved conflict, see §2.2 and §7.
- PSG-JEPA: arXiv 2608.06799 — Yan et al., Aug 2026 preprint. *Is Forward Prediction Enough? Physical State Grounding for JEPA World Models.*
- Ground-JEPA: Research Square rs-10511499, posted 29 Jul 2026, not peer-reviewed. *Learning Physically Grounded Latent World Models for Zero-Shot Dynamics Generalization of Legged Robots.*
- LeWM: arXiv 2603.19312 — Maes et al. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.*

**Not yet independently verified:** GrandTour's Zarr contents and actual ANYmal policy rate; JEPA-WM paper Appendices E–G; the full V-JEPA 2.1 and Ground-JEPA paper bodies beyond their abstracts/BibTeX; the V-JEPA 2.1 weights' license.
# ADR 0001: Slurm GPU allocation is unusable on `iisc`; select GPUs by free VRAM

- Status: Accepted (workaround; proper fix is cluster-side)
- Date: 2026-09-29
- Affects: `scripts/slurm/jepa_baseline.sbatch`, `scripts/slurm/jepa_smoke.sbatch`

## Context

Jobs on the `iisc` node were failing at (or shortly after) startup. The failures
took three different shapes, which initially looked unrelated:

- `RuntimeError: GPU 1 (rank 1) already has 147.7 GiB in use by another process` (job 1383, exit 1)
- `RuntimeError: training requires an allocated CUDA/ROCm GPU` (jobs 1410, 1411 — `torch.cuda.is_available()` was `False`)
- `Memory access fault by GPU node-2 ... on address (nil)` after ~3 h of training (job 1384, SIGABRT)
- `RuntimeError: GPU 0 (rank 0) already has 164.6 GiB in use` (job 1413)

All of these trace back to one thing: **`SLURM_JOB_GPUS` on this cluster is not a
physical GPU index, so pinning `ROCR_VISIBLE_DEVICES` to it lands jobs on the
wrong (and often occupied) GPU.** This ADR records the evidence and the
workaround.

Node facts: Slurm 23.11.4, 8× AMD Instinct MI300X (`gfx942`), 192 GiB HBM each.
`cgroup.conf` has `ConstrainDevices=yes`; `slurm.conf` uses
`SelectType=select/cons_tres`, `TaskPlugin=task/cgroup`, `GresTypes=gpu`,
`NodeName=iisc ... Gres=gpu:8`.

## Findings

### 1. `gres.conf` enumerates the XCD partitions of one card as 8 "GPUs"

```
NodeName=iisc Name=gpu Type=mi300x File=/dev/dri/renderD128
NodeName=iisc Name=gpu Type=mi300x File=/dev/dri/renderD129
...
NodeName=iisc Name=gpu Type=mi300x File=/dev/dri/renderD135
```

The eight `renderD` nodes map to hardware like this (from
`/sys/class/drm/renderD*/device`):

| renderD | device | meaning |
|---|---|---|
| 128 | `0000:1b:00.0` | **physical GPU 0** (whole OAM) |
| 129–135 | `amdgpu_xcp_0..6` | GPU 0's 7 XCD partitions |
| 136 | `0000:3d:00.0` | physical GPU 1 |
| 137–143 | `amdgpu_xcp_7..13` | GPU 1's XCD partitions |
| 144 | `0000:4e:00.0` | physical GPU 2 |
| 152 | `0000:5f:00.0` | physical GPU 3 |
| 160 | `0000:9d:00.0` | physical GPU 4 |
| 168 | `0000:bd:00.0` | physical GPU 5 |
| 176 | `0000:cd:00.0` | physical GPU 6 |
| 184 | `0000:dd:00.0` | physical GPU 7 |

There are 64 `renderD` nodes total = 8 physical GPUs × (1 main + 7 XCD) each.

So Slurm's entire "8 GPU" pool is **physical GPU 0 plus 7 of its XCD
partitions**. Physical GPUs 1–7 are not in Slurm's pool at all — which is why
other users' vLLM servers run on them completely outside Slurm's accounting.

### 2. Slurm hides physical GPU 0 from the job's runtime

A diagnostic job opened every `/dev/dri/renderD*` node; the host shell did the
same. Only the 8 "main" nodes are openable (the `amdgpu_xcp_*` nodes are not
standalone devices):

```
host shell : OK=8  DENY=56   (renderD128,136,144,152,160,168,176,184)
Slurm job  : OK=7  DENY=57   (the same minus renderD128)
```

Inside a Slurm job, **physical GPU 0's main render node is denied**, so the ROCm
runtime enumerates only physical GPUs **1–7, renumbered 0–6**. `rocm-smi`
inside the job shows the same 7-GPU, shifted view — its values match the host's
`GPU[i+1]`, including identical `Memory Activity` counters (e.g. `848006462`
appears on host `GPU[7]` and job `GPU[6]`). `torch.cuda.device_count()` inside
the job is 7 **even with all visibility variables cleared**, so this is not an
env-var artifact — it is the device set the job can actually see.

### 3. `SLURM_JOB_GPUS` is a gres slot, not a physical index

A running job's environment (`/proc/<pid>/environ`):

```
SLURM_JOB_GPUS=7   SLURM_GTIDS=0
ROCR_VISIBLE_DEVICES=0   CUDA_VISIBLE_DEVICES=0
```

Slurm's gres plugin populates `ROCR_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` with
the *job-local* index (`SLURM_GTIDS`), assuming the device cgroup will remap it
to the allocated device. On this node that remap does not happen (the devices
cgroup reads `a *:* rwm` for jobs, and AMD isolation via a single shared
`/dev/kfd` does not restrict enumeration anyway). Two consequences:

- The job-local index and the physical index diverge by one (physical GPU 0 is
  hidden), so `ROCR_VISIBLE_DEVICES=$SLURM_JOB_GPUS` selects `physical GPU N+1`.
- `SLURM_JOB_GPUS` can be `7` while the job sees only indices `0–6`, so the pin
  is out of range and `torch.cuda.is_available()` returns `False`.

Additionally, `ROCR_VISIBLE_DEVICES` is the low-level filter: setting it *and*
`HIP/CUDA_VISIBLE_DEVICES` to the same index double-filters (ROCR narrows to one
device, then HIP asks for an out-of-range 8th), also yielding zero devices:

```
ROCR_VISIBLE_DEVICES=7                        -> avail True,  1 device
ROCR_VISIBLE_DEVICES=7 HIP_VISIBLE_DEVICES=7  -> avail False, 0 devices
ROCR_VISIBLE_DEVICES=7 HIP_VISIBLE_DEVICES=0  -> avail True,  1 device
```

### 4. The observed failures, mapped

| Job | requested | `SLURM_JOB_GPUS` | outcome |
|---|---|---|---|
| 1383 | 2 | 3,4 | landed on physical GPU 1 (vLLM, 147.7 GiB) → guardrail |
| 1384 | 1 | 3 | ran 3 h on physical GPU 0, then GPU memory-access fault (SIGABRT) |
| 1410 | 1 | 7 | `ROCR=7 HIP=7` → 0 devices → "no allocated GPU" |
| 1411 | 1 | 7 | `ROCR=7` alone → index 7 out of range (job sees 0–6) → 0 devices |
| 1413 | 1 | 3 | `ROCR=3` → physical GPU 4 (164.6 GiB) → guardrail |
| 1416 | 4 | 3,4,5,7 | `ROCR=3,4,5,7` includes out-of-range 7 |

The host's GPU load is also very dynamic: at one point GPUs 3–7 were free while
0/1 were busy; minutes later the reverse. Any static pin is therefore a gamble.

## Decision

Stop trusting `SLURM_JOB_GPUS`. In both sbatch scripts, clear the visibility
variables, enumerate the GPUs the job can actually see, and pin
`ROCR_VISIBLE_DEVICES` to the freest one(s) by `torch.cuda.mem_get_info`:

```bash
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES="$(
  GPUS="${SLURM_GPUS_ON_NODE:-1}" "$PROJECT_DIR/.venv/bin/python3" - <<'PY'
import os, torch
want = int(os.environ.get("GPUS", "1"))
ranked = sorted(range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.mem_get_info(i)[0], reverse=True)
print(",".join(map(str, sorted(ranked[:want]))))
PY
)"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

This works in the job's own (shifted) index space, so it is independent of the
broken slot mapping and of GPU 0 being hidden. The `min_free_gpu_fraction`
guard in `_distributed` stays as a fail-fast backstop.

## Verification

- Job 1417 (`--gres=gpu:1`): `SLURM_JOB_GPUS=3` → `ROCR_VISIBLE_DEVICES=2`,
  `free_gib=191.5`, guardrail passed, training started.
- Job 1418 (`--gres=gpu:4`): `SLURM_JOB_GPUS=3,4,5,6` →
  `ROCR_VISIBLE_DEVICES=2,3,4,5`, all four ranks reported ~190 GiB free,
  `world_size=4`, running.

## Consequences and ceiling

This is a **workaround**, with a known ceiling:

- It picks the freest GPU **at job start**. Since AMD device isolation is not
  actually enforced on this cluster, a non-Slurm process (e.g. someone's vLLM
  server) can still take memory on the chosen GPU afterwards. The
  `min_free_gpu_fraction` guard catches the startup case only; a mid-run fault
  (job 1384) remains possible. The real fix is cluster-side.
- It adds one `torch` import (~seconds) to every job start.

**Proper cluster-side fix (needs admin):**

1. `gres.conf` should map the 8 *physical* main render nodes as the 8 GPUs:
   `renderD128, 136, 144, 152, 160, 168, 176, 184`.
2. AMD device isolation must actually work (the devices cgroup is currently
   allow-all for jobs, and `/dev/kfd` is shared), otherwise GPU allocation stays
   advisory and non-Slurm workloads cannot be excluded.
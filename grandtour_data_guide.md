# Consuming the GrandTour Dataset (Hugging Face)

**Dataset:** `leggedrobotics/grand_tour_dataset` on Hugging Face
**Paper:** Frey, Tuna, Fu, Patterson, Xu, Fallon, Cadena, Hutter. *GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation.* arXiv 2602.18164.

---

## 1. Access

1. Register via the official webpage (https://grand-tour.leggedrobotics.com).
2. Accept the gated dataset terms on the Hugging Face page.
3. `pip install huggingface_hub` and `huggingface-cli login` (interactive token prompt — if your environment can't accept stdin, run this in a local terminal rather than inside a notebook cell).

---

## 2. Downloading and extracting

The dataset is **not** distributed as raw `.zarr`/`.jpeg` files directly on the Hugging Face repo. To avoid checking in more than a million individual files, every topic-per-mission is packed into its own `.tar`. You download the tars, then extract them yourself.

### 2.1 Download

```python
from huggingface_hub import snapshot_download

mission = "2024-11-02-17-18-32"

# Full mission, every topic:
allow_patterns = [f"{mission}/*"]

# Or a single topic plus the mission's YAML metadata:
# topic = "alphasense_front_center"
# allow_patterns = [f"{mission}/*{topic}*", f"{mission}/*.yaml"]

hugging_face_data_cache_path = snapshot_download(
    repo_id="leggedrobotics/grand_tour_dataset",
    allow_patterns=allow_patterns,
    repo_type="dataset",
)
```

`snapshot_download` caches to `~/.cache/huggingface/...` and resumes automatically if interrupted — re-running the same call after a failure does not re-download completed files.

### 2.2 Extraction

Downloaded files land as `.tar` per topic (e.g. `hdr_front.tar`, `anymal_state_actuator.tar`) plus loose `.yaml` metadata files. Extract to a location **outside** the Hugging Face cache directory:

```python
from pathlib import Path
import tarfile, shutil, re

dataset_folder = Path("~/grand_tour_dataset").expanduser()
dataset_folder.mkdir(parents=True, exist_ok=True)

def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    def to_regex(patterns):
        parts = [f".*{re.escape(p).replace(r'\*','.*').replace(r'\?','.')}$" for p in patterns]
        return re.compile("|".join(parts))
    pattern = to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]

    for f in [x for x in files if x.suffix == ".tar"]:
        dest = dataset_folder / f.relative_to(cache)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(f, "r") as tar:
            tar.extractall(path=dest.parent)

    for f in [x for x in files if x.suffix != ".tar" and x.is_file()]:
        dest = dataset_folder / f.relative_to(cache)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

move_dataset(hugging_face_data_cache_path, dataset_folder, allow_patterns=allow_patterns)
```

Result, per mission, on disk:

```
~/grand_tour_dataset/<mission>/
├── data/          ← Zarr store (opened as a group; holds every topic's arrays)
├── images/        ← <topic>/<image_id:06d>.jpeg  or  .png  (raw pixel files, not in Zarr)
└── metadata/      ← <topic>.yaml (per-topic calibration/frame_id/description)
```

### 2.3 Opening the Zarr store

```python
import zarr
mission_root = zarr.open_group(store=dataset_folder / mission / "data", mode="r")
list(mission_root.keys())   # every topic in this mission
```

Tutorials were tested against `zarr==3.0.7`.

---

## 3. Full topic taxonomy

### 3.1 Cameras — RGB

| Topic | Rate / resolution | Sensor |
|---|---|---|
| `hdr_front`, `hdr_left`, `hdr_right` | 10 Hz, 1920×1280 | TierIV C1 (120 dB HDR, color, 2.5 MP) — Boxi payload |
| `alphasense_front_center`, `alphasense_left`, `alphasense_right` | 10 Hz, 1440×1080 | Sevensense Alphasense/CoreResearch (Sony IMX-273, color, 1.6 MP) — Boxi payload |
| `zed2i_left_images`, `zed2i_right_images` | 15 Hz, 1920×1080 | Stereolabs ZED2i (color, 4 MP; mounted upside-down — left/right naming is flipped relative to the ZED2i driver's own convention) |

### 3.2 Cameras — mono

| Topic | Rate / resolution | Sensor |
|---|---|---|
| `alphasense_front_left`, `alphasense_front_right` | 10 Hz, 1440×1080 | Sevensense Alphasense (mono, 1.6 MP) |

### 3.3 Cameras — depth (what Track 1/2 need)

| Topic | Rate / resolution | Sensor |
|---|---|---|
| `depth_camera_front_upper`, `depth_camera_front_lower`, `depth_camera_rear_upper`, `depth_camera_rear_lower`, `depth_camera_left`, `depth_camera_right` | 15 Hz, 848×480 | Intel RealSense D435i — integrated **directly in the ANYmal D**, not the Boxi payload |
| `zed2i_depth_image` | 15 Hz, 1920×1080 | Stereolabs ZED2i, SDK-generated. **Uses different intrinsics from `zed2i_right_images`** even though it's derived from that camera — do not reuse one camera's K matrix for the other. |
| `zed2i_depth_confidence_image` | 15 Hz, 1920×1080 | Per-pixel confidence map for the above |

**The join-key gotcha:** the Zarr array for every image and depth-image topic holds only `sequence_id` and `timestamp` — never pixel data. Pixel data lives separately as files under `images/<topic>/`. And critically:

> `sequence_id != image_id`. `sequence_id` is the original ROS message timestamp-derived ID from data capture (not guaranteed to start at 0). `image_id` is a zero-indexed counter assigned during the Zarr/PNG conversion, and it's what indexes both the Zarr array position **and** the filename.

Concretely, to load frame `i` of `hdr_front`:

```python
image_id = i                      # zero-indexed position, matches array index
image_path = mission_folder / "images" / "hdr_front" / f"{image_id:06d}.jpeg"
```

**Not** `mission_root["hdr_front"]["sequence_id"][i]` as a filename component — that value is the ROS-runtime ID, not the file index. A dataloader keyed off `sequence_id` will silently misalign frames on any mission where capture didn't start at ID 0.

**Depth pixel format:** RealSense and ZED2i depth PNGs are 16-bit, storing depth in millimeters. Convert to meters:

```python
depth_m = imageio.imread(depth_path).astype(np.float32) / 1000.0
```

**Three more things that matter for your encoder design:**
- **Camera calibration — both intrinsics (`K`, `D`) and extrinsics (`transform`) — changes between missions.** Reload per-mission from `metadata/<topic>.yaml` or `mission_root[topic].attrs["camera_info"]` / `.attrs["transform"]`. Do not cache one mission's calibration and reuse it for another.
- **Self-occlusion is documented directly, not just a theoretical concern:** the depth cameras integrated in the ANYmal D are tilted downward toward the ground, and the left and right portions of the field of view are obstructed by the robot's own legs. This is directly relevant to Track 1 — it's exactly the occlusion/sparsity condition JEPLO's latent is supposed to be robust to. Worth a quick plot of a few depth frames at a typical gait phase before training, and a natural connection to testing that robustness claim directly in E1.8 rather than assuming it.
- **Camera distortion model is `equidistant` (fisheye)**, not standard radial-tangential — confirmed on `hdr_front`. If you reproject LiDAR onto any camera (Track 1 LiDAR stretch goal, or any camera-LiDAR fusion), use `cv2.fisheye.projectPoints`, not `cv2.projectPoints`. The wrong projection model produces an overlay that looks plausible and is geometrically wrong.

### 3.4 LiDAR

| Topic | Rate | Sensor |
|---|---|---|
| `hesai`, `hesai_undistorted` | 10 Hz | Hesai XT-32 |
| `velodyne`, `velodyne_undist` | 10 Hz | Velodyne VLP-16 — integrated in the ANYmal D itself |
| `livox_points`, `livox_points_undistorted` | 10 Hz | Livox Mid360 (mounted upside-down, non-repetitive scan pattern) |
| `dlio_hesai_points_undistorted` | 10 Hz | Hesai points, motion-undistorted specifically via Direct LiDAR Inertial Odometry rather than leg-inertial odometry |

The `*_undistorted` variants are motion-compensated using the leg-inertial odometry estimate (except `dlio_hesai_points_undistorted`, which uses DLIO instead). For a world model conditioned on point clouds, use the undistorted variant — raw scans carry motion distortion from the platform's own movement during the scan window, which is nuisance variation you don't want the encoder learning to model.

**Padding gotcha:** LiDAR scans have a variable point count per scan, but Zarr arrays require fixed shape. Point clouds are stored padded to a fixed size with a companion `valid` field giving the true count:

```python
n_valid = int(mission_root[lidar_tag]["valid"][idx])
points = mission_root[lidar_tag]["points"][idx, :n_valid, :]
```

Using the full padded array without this slice includes garbage padding points in every downstream computation.

### 3.5 IMUs

| Topic | Rate | Sensor |
|---|---|---|
| `anymal_imu` | 400 Hz | Epson — integrated in ANYmal D |
| `adis_imu` | 200 Hz | Analog Devices ADIS16475-2 |
| `stim320_imu` (+ `stim320_gyroscope_temperature`, `stim320_accelerometer_temperature` at 500 Hz) | 500 Hz | Safran STIM320 |
| `alphasense_imu` | 200 Hz | Bosch BMI085 — integrated in the Alphasense camera unit |
| `livox_imu` | 200 Hz | TDK ICM40609 — integrated in the Livox Mid360 |
| `ap20_imu` | 200 Hz | Leica AP20 — **only populated while Bluetooth to the MS60 total station is active** |
| `cpt7_imu` | 100 Hz | Honeywell HG4930 — integrated in the Novatel CPT7 |

`anymal_imu` at 400 Hz is the natural proprioceptive-rate reference if you need IMU-rate supervision rather than the `anymal_state_*` topics (confirm their actual rate directly — see §5).

---

## 4. Proprioceptive / state topics — mapped onto the shared evaluation protocol

`anymal_state_state_estimator` already contains essentially the entire $s_t$ probe target vector defined in the shared evaluation protocol.

### 4.1 `anymal_state_actuator` — per-joint actuator state

Fields exist per joint, indexed `00` through `11` (12 joints total, matching ANYmal D's 12 actuated DoF), each with: `command_current`, `command_joint_torque`, `command_mode`, `command_pid_gains_{d,i,p}`, `command_position`, `command_velocity`, `state_current`, `state_gear_position`, `state_gear_velocity`, `state_imu_{ang_vel, lin_acc, orien}` (+ covariances), `state_joint_{acceleration, position, torque, velocity}`, `state_statusword`.

**Confirmed joint index ordering — use this exact order everywhere in your pipeline** to stay consistent between GrandTour data and your Isaac Lab environment:

```
0: LF_HAA   1: LF_HFE   2: LF_KFE
3: RF_HAA   4: RF_HFE   5: RF_KFE
6: LH_HAA   7: LH_HFE   8: LH_KFE
9: RH_HAA   10: RH_HFE  11: RH_KFE
```

(`HAA` = hip abduction/adduction, `HFE` = hip flexion/extension, `KFE` = knee flexion/extension — standard ANYmal naming.) A mismatch here between your simulator's joint order and GrandTour's would silently corrupt every joint-space loss in Tracks 1–3.

### 4.2 `anymal_state_state_estimator` — the full physical state vector

Fields: `{LF,RF,LH,RH}_FOOT_{contact, friction_coef, normal, restitution_coef, state, wrench_force, wrench_torque}`, `joint_accelerations`, `joint_efforts`, `joint_positions`, `joint_velocities`, `pose_orien`, `pose_pos`, `twist_ang`, `twist_lin`.

Direct mapping onto the shared evaluation protocol's probe target $s_t$:

| Protocol symbol | GrandTour field |
|---|---|
| $p_t^{\text{base}}$ | `pose_pos` |
| $v_t^{\text{base}}$ | `twist_lin` |
| $\omega_t^{\text{base}}$ | `twist_ang` |
| $g_t^{\text{base}}$ | derive from `pose_orien` (rotate the world gravity vector into the base frame) |
| $q_t$ | `joint_positions` (12-dim, ordering per §4.1) |
| $\dot q_t$ | `joint_velocities` |
| $c_t$ | `{LF,RF,LH,RH}_FOOT_contact` |

The probe target vector doesn't need to be assembled by hand from multiple heterogeneous sources — it's essentially one topic, plus a gravity-vector rotation.

### 4.3 `anymal_state_odometry` vs `anymal_state_state_estimator`

Don't conflate these. `anymal_state_odometry` is specifically ANYmal's leg-inertial odometry solution (pose/twist only) — it drifts over time like any dead-reckoning estimate. `anymal_state_state_estimator` is the fuller state estimate including contact and per-joint dynamics. Use `state_estimator` as your probe target; treat `odometry` as one specific (lower-fidelity) trajectory estimate, useful mainly as a comparison point for how much better your world model's implicit state tracking is doing.

### 4.4 `anymal_command_twist`

Operator-issued linear/angular twist command. This is a joystick-level velocity command, not the low-level joint action your RL policy will output — but it's a legitimate source of $v_{xy}^{\text{cmd}}, \omega_z^{\text{cmd}}$ if you want to condition a world model on real commanded velocities for the real-data evaluation, rather than only replaying logged joint actions.

---

## 5. Ground-truth / localization precision hierarchy

Confirmed topic names, from highest precision to lowest:

| Topic | Precision | Availability |
|---|---|---|
| `prism_position` | Sub-cm | Only while direct line-of-sight between Boxi and the Leica MS60 total station is maintained — position only, not full pose |
| `cpt7_ie_tc_odometry` / `cpt7_ie_tc_tf` | Highest among GNSS-based solutions (tightly-coupled) | Outdoor, GNSS-dependent |
| `cpt7_ie_rt_odometry` / `cpt7_ie_rt_tf` | Lower (real-time PPP) | Outdoor, GNSS-dependent |
| `gnss_raw_cpt7_ie_{rt,tc}`, `navsatfix_cpt7_ie_tc` | Raw GNSS solutions underlying the above | Outdoor |
| `dlio_map_odometry` / `dlio_hesai_points_undistorted` | LiDAR-inertial, works indoors | Most missions |
| `zed2i_vio_map` | Visual-inertial | Most missions |
| `anymal_state_odometry` | Leg-inertial, drifts | All missions |

For the shared evaluation protocol's real-data open-loop evaluation, use `anymal_state_state_estimator` as your primary comparison target since it's what a deployed policy would actually have access to, but pull `prism_position` or `cpt7_ie_tc_odometry` where available as an independent accuracy check on how much drift the state estimator itself carries on a given mission — this directly informs the noise-floor caveat already in the shared protocol.

---

## 6. Mission structure and splits

**Mission identifiers:** `YYYY-MM-DD-hh-mm-ss` timestamp, each also carrying a human-readable tag (e.g. `ETH-1`). All missions carry the same fields **except where physically impossible to collect** — the explicit example given is GNSS indoors. In practice the core onboard sensors (cameras, depth, LiDAR, proprioception, most IMUs) are consistently present; it's specifically the GNSS/total-station position chain that's environment-dependent.

**The dataset ships its own train/val/test split.** Use it as your base rather than building one from scratch:

- **Train and val:** all data available, including the high-precision position sources.
- **Test:** accurate position data — Leica MS60 total station and Novatel CPT7 GNSS/IMU — is deliberately removed, so it's specifically missing the ground truth you'd want for a real-data evaluation.

**Practical implication:** since your real-data evaluation wants the most accurate available reference trajectories, prefer **train/val missions** for anything where you need `prism_position` or `cpt7_ie_tc_odometry` as ground truth. Build your own held-out split for evaluation **within** the train/val pool, at the mission level (never frame level, since consecutive frames within a mission are near-duplicates and would leak across the split) — the dataset's own "test" split isn't usable for this purpose, since it's specifically missing the ground truth you'd evaluate against.

---

## 7. Time synchronization

The dataset gives you per-topic `timestamp` arrays, not a pre-aligned tensor — you build the alignment yourself.

- **Pick the slowest relevant sensor as the base clock.** Depth cameras at 15 Hz are the natural choice for the depth-based recipes; upsample proprioception to it, not the other way around, since proprioception can be validly interpolated but a depth frame cannot be invented between two real frames.
- **For proprioceptive channels** (joint positions, velocities, base velocity), linear interpolation between the two nearest timestamped samples is standard and defensible.
- **For orientation (quaternions)**, use SLERP, not linear interpolation — linearly interpolating quaternion components and renormalizing introduces a small but systematic bias, and both $g_t^{\text{base}}$ (projected gravity) and any grounding-head targets depend on getting this right.
- **Set an explicit maximum gap threshold.** If the nearest proprioceptive sample is more than, say, 50 ms from the target depth timestamp, drop that frame rather than interpolating across it — a large gap usually means a dropped message, not a valid interpolation window.
- **Confirmed rates to build the sync plan around:** depth cameras 15 Hz, `hdr_front`/`alphasense_*` 10 Hz, all LiDAR 10 Hz, `anymal_imu` 400 Hz. The `anymal_state_actuator` / `anymal_state_state_estimator` rate is not stated in the documentation reviewed here — check the actual `timestamp` array spacing for your downloaded mission directly (`np.diff(mission_root["anymal_state_state_estimator"]["timestamp"][:])`) before assuming a control-loop rate.

---

## 8. Concrete next steps

1. Register, accept the HF gate, `huggingface-cli login`.
2. Download **one mission** with `snapshot_download` + the extraction helper in §2.
3. Open it with `zarr.open_group`, confirm the topic list matches §3–§4.
4. Build the frame loader for one depth topic and one proprioceptive topic, using `image_id` (not `sequence_id`) as the join key — verify this explicitly on a few frames before trusting it at scale.
5. Check the actual `anymal_state_state_estimator` timestamp spacing for that mission (§7) rather than assuming a rate.
6. Confirm the joint ordering in `anymal_state_actuator` matches what you feed into Isaac Lab (§4.1) — mismatch here is silent and will corrupt every joint-space loss.
7. Plot a handful of depth frames and check leg self-occlusion at a typical gait phase before committing to the depth-only Track 1/2 recipe as-is.
8. Only after all of the above are validated on one mission, move to the multi-mission download and the train/val split from §6.

---

## 9. References

- Frey, Tuna, Fu, Patterson, Xu, Fallon, Cadena, Hutter. *GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation.* arXiv 2602.18164.
- Tuna, Frey, Fu, Weibel, Patterson, Krummenacher, Müller, Nubert, Fallon, Cadena, Hutter. *Boxi: Design Decisions in the Context of Algorithmic Performance for Robotics.* RSS 2025.
- `examples_hugging_face/notebooks/access.ipynb` and `explore.ipynb`, `leggedrobotics/grand_tour_dataset`.
- Dataset card: https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset
- Official page: https://grand-tour.leggedrobotics.com
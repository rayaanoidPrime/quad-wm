"""MuJoCo implementation of the Simulator contract (docs/adr/0003).

Loads a robot-only MJCF, adds the terrain heightfield, lighting and the
front camera through ``MjSpec``, and steps it at the GrandTour action rate.
Rendering needs a GL backend; set ``MUJOCO_GL=egl`` on headless nodes.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..data.grandtour import FEET, JOINT_ORDER, project_gravity
from .base import DynamicsSpec, SimState, SimulatorUnavailable, TerrainSpec
from .terrain import terrain_heights

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    mujoco = None
    _IMPORT_ERROR = exc


def _camera_quat(pitch_deg: float) -> list[float]:
    """MuJoCo cameras look down their -z axis; aim it along body +x, pitched down."""
    pitch = np.radians(pitch_deg)
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    x_axis, y_axis = np.array([0.0, -1.0, 0.0]), np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    x, y, z, w = Rotation.from_matrix(np.column_stack((x_axis, y_axis, -forward))).as_quat()
    return [w, x, y, z]


class MujocoSimulator:
    def __init__(self, config: dict):
        if mujoco is None:
            raise SimulatorUnavailable(f"mujoco is not installed (uv sync --extra sim): {_IMPORT_ERROR}")
        mjcf = Path(config["mjcf"])
        if not mjcf.is_file():
            raise SimulatorUnavailable(f"robot MJCF not found at {mjcf}; see configs/sim/ for setup")
        self.config = config
        self.tick_hz = float(config["tick_hz"])
        self.control_hz = float(config["control_hz"])
        self.frames_per_tick = round(self.control_hz / self.tick_hz)
        self.default_pose = np.asarray(config["default_joint_pos"], dtype=np.float64)
        self._base_spec = mujoco.MjSpec.from_file(str(mjcf))
        self.model = self.data = self.renderer = None

    # -- episode setup -------------------------------------------------------

    def _compile(self, terrain: TerrainSpec, seed: int) -> None:
        spec = self._base_spec.copy()
        cfg, t = self.config, self.config["terrain"]
        heights = terrain_heights(
            terrain, length_m=t["length_m"], width_m=t["width_m"], cell_m=t["cell_m"], seed=seed
        )
        low, span = heights.min(), max(np.ptp(heights), 1e-3)
        spec.add_hfield(
            name="terrain",
            size=[t["length_m"] / 2, t["width_m"] / 2, span, 0.5],
            nrow=heights.shape[0],
            ncol=heights.shape[1],
            userdata=((heights - low) / span).ravel().tolist(),
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain", pos=[0, 0, low], rgba=[0.5, 0.5, 0.45, 1]
        )
        spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        cam = cfg["camera"]
        spec.body(cfg["base_body"]).add_camera(
            name="front", pos=cam["pos"], quat=_camera_quat(cam["pitch_deg"]), fovy=cam["fovy_deg"]
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.heights, self.cell_m = heights, t["cell_m"]
        self.substeps = round(1.0 / (self.control_hz * self.model.opt.timestep))

        model = self.model
        self.base_id = model.body(cfg["base_body"]).id
        self.qpos_ids = np.array([model.joint(name).qposadr[0] for name in JOINT_ORDER])
        self.qvel_ids = np.array([model.joint(name).dofadr[0] for name in JOINT_ORDER])
        self.actuator_ids = np.array([model.actuator(name).id for name in JOINT_ORDER])
        # Foot = the sphere collision geom on each shank.
        self.foot_geoms = [
            set(np.flatnonzero(
                (model.geom_bodyid == model.body(f"{foot}_SHANK").id)
                & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            ))
            for foot in FEET
        ]
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(model, cam["height"], cam["width"]) if cam["modalities"] else None

    def _apply_dynamics(self, dynamics: DynamicsSpec) -> None:
        model = self.model
        model.body_mass[self.base_id] *= dynamics.base_mass_scale
        model.geom_friction[:, 0] *= dynamics.friction_scale
        kp, kd = self.config["kp"], self.config["kd"]
        model.actuator_gainprm[self.actuator_ids, 0] = kp
        model.actuator_biasprm[self.actuator_ids, 1:3] = [-kp, -kd]
        delay = round(dynamics.latency_ms / 1000.0 / model.opt.timestep)
        self.pending = deque([self.default_pose.copy()] * (delay + 1), maxlen=delay + 1)
        self.dynamics = dynamics

    def reset(self, *, seed: int, terrain: TerrainSpec = TerrainSpec(), dynamics: DynamicsSpec = DynamicsSpec()):
        self._compile(terrain, seed)
        self._apply_dynamics(dynamics)
        start_x = -self.config["terrain"]["length_m"] / 2 + 1.0
        self.data.qpos[:3] = [start_x, 0.0, self.config["spawn_height"]]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # facing +x, down the course
        self.data.qpos[self.qpos_ids] = self.default_pose
        self.data.ctrl[self.actuator_ids] = self.default_pose
        mujoco.mj_forward(self.model, self.data)
        self.terrain = terrain
        return self._observe()

    # -- stepping ------------------------------------------------------------

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        targets = np.asarray(action, dtype=np.float64).reshape(self.frames_per_tick, len(JOINT_ORDER))
        for target in targets:
            for _ in range(self.substeps):
                # Latency: the actuator sees the target issued `delay` physics steps ago.
                self.pending.append(target)
                self.data.ctrl[self.actuator_ids] = self.pending[0]
                mujoco.mj_step(self.model, self.data)
        return self._observe()

    def get_state(self) -> SimState:
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, physics, spec)
        return SimState(physics, {"pending": [target.copy() for target in self.pending]})

    def set_state(self, state: SimState) -> None:
        mujoco.mj_setState(self.model, self.data, state.physics, mujoco.mjtState.mjSTATE_INTEGRATION)
        self.pending = deque(state.extra["pending"], maxlen=self.pending.maxlen)
        mujoco.mj_forward(self.model, self.data)

    # -- observation ---------------------------------------------------------

    def _observe(self) -> dict[str, np.ndarray]:
        data = self.data
        w, x, y, z = data.qpos[3:7]
        orientation = np.array([x, y, z, w])  # GrandTour order (x, y, z, w)
        rotation = data.xmat[self.base_id].reshape(3, 3)
        touching = set(data.contact.geom1[: data.ncon]) | set(data.contact.geom2[: data.ncon])
        state = {
            "pose_pos": data.qpos[:3].copy(),
            "orientation": orientation,
            "lin_vel": rotation.T @ data.qvel[:3],  # freejoint linear velocity is world frame
            "ang_vel": data.qvel[3:6].copy(),  # freejoint angular velocity is base frame
            "gravity": project_gravity(orientation),
            "joint_pos": data.qpos[self.qpos_ids].copy(),
            "joint_vel": data.qvel[self.qvel_ids].copy(),
            "contacts": np.array([float(bool(feet & touching)) for feet in self.foot_geoms]),
        }
        state = {key: value.astype(np.float32) for key, value in state.items()}
        state["proprio"] = np.concatenate(
            [state[key] for key in ("lin_vel", "ang_vel", "gravity", "joint_pos", "joint_vel")]
        )
        state |= self._privileged()
        if self.renderer is not None:
            self.renderer.update_scene(data, camera="front")
            modalities = self.config["camera"]["modalities"]
            if "rgb" in modalities:
                state["rgb"] = self.renderer.render().copy()
            if "depth" in modalities:
                self.renderer.enable_depth_rendering()
                state["depth"] = self.renderer.render().copy()
                self.renderer.disable_depth_rendering()
        return state

    def _privileged(self) -> dict[str, np.ndarray]:
        """Probe-only labels (never a training input; baseline recipe §2.1)."""
        rows, cols = self.heights.shape
        x, y = self.data.qpos[:2]
        col = int(np.clip((x + self.config["terrain"]["length_m"] / 2) / self.cell_m, 0, cols - 1))
        row = int(np.clip((y + self.config["terrain"]["width_m"] / 2) / self.cell_m, 0, rows - 1))
        return {
            "terrain_height": np.float32(self.heights[row, col]),
            "friction_scale": np.float32(self.dynamics.friction_scale),
        }

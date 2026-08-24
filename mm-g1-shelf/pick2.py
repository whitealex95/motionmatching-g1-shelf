"""Retarget the Y-up, 34-joint G1 representation in ``pick2.npz``.

``pick2.npz`` is not a MuJoCo qpos clip.  It stores posed joint positions and
global rotations in a Y-up convention.  The joint positions correspond to the
G1 links in this project, so each frame is fitted to the MuJoCo G1's 29 hinge
joints while preserving its free-root pose.  The fit is typically below
0.1 mm RMS.
"""
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import least_squares

import config as C
import quat


FPS = 30.0
REACH_START = 28
GRASP_FRAME = 54
LIFT_START = 56
HOLD_START = 85
MAX_RMS_ERROR = 3e-4

# pick2 joint index -> MuJoCo body ID.  pick2 has end-effectors after each
# ankle and wrist; the menagerie G1 represents those as geoms, not bodies.
BODY_TO_PICK2 = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
    14: 15, 15: 16, 16: 17,
    17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24,
    24: 26, 25: 27, 26: 28, 27: 29, 28: 30, 29: 31, 30: 32,
}
BODY_IDS = np.array(list(BODY_TO_PICK2), dtype=np.int32)
PICK2_IDS = np.array(list(BODY_TO_PICK2.values()), dtype=np.int32)

# pick2 axes are (right, up, forward); MuJoCo uses (forward, left, up).
SOURCE_TO_MUJOCO = np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])


def _joint_limits(model):
    lower = np.full(29, -np.inf)
    upper = np.full(29, np.inf)
    for joint_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        if 7 <= qadr < 36 and model.jnt_limited[joint_id]:
            lower[qadr - 7], upper[qadr - 7] = model.jnt_range[joint_id]
    return lower, upper


def _phase_and_contact(nframes):
    if nframes <= HOLD_START:
        raise ValueError(f"pick2 needs more than {HOLD_START} frames; got {nframes}")
    phase = np.full(nframes, C.PHASE_HOLD, dtype=np.int32)
    phase[:REACH_START] = C.PHASE_IDLE
    phase[REACH_START:GRASP_FRAME] = C.PHASE_REACH
    phase[GRASP_FRAME:LIFT_START] = C.PHASE_GRASP
    phase[LIFT_START:HOLD_START] = C.PHASE_LIFT
    contact = np.zeros(nframes, dtype=np.float64)
    contact[GRASP_FRAME:] = 1.0
    return phase, contact


def retarget(source_path):
    """Return ``(qpos, phase, contact)`` generated from a ``pick2.npz`` file."""
    source_path = Path(source_path)
    with np.load(source_path) as source:
        required = {"posed_joints", "global_rot_mats", "root_positions"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"{source_path} is missing keys: {sorted(missing)}")
        posed = source["posed_joints"].astype(np.float64)
        root_positions = source["root_positions"].astype(np.float64)
        root_rotations = source["global_rot_mats"][:, 0].astype(np.float64)

    if posed.ndim != 3 or posed.shape[1:] != (34, 3):
        raise ValueError(f"Expected posed_joints shaped (T, 34, 3), got {posed.shape}")
    if root_positions.shape != (len(posed), 3):
        raise ValueError("root_positions must be shaped (T, 3)")
    if root_rotations.shape != (len(posed), 3, 3):
        raise ValueError("global_rot_mats must be shaped (T, 34, 3, 3)")

    model = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    data = mujoco.MjData(model)
    lower, upper = _joint_limits(model)
    previous = np.zeros(29)
    qpos = np.empty((len(posed), 36), dtype=np.float64)
    rms_errors = []

    for frame in range(len(posed)):
        base = np.zeros(36)
        base[:3] = SOURCE_TO_MUJOCO @ root_positions[frame]
        root_rotation = (SOURCE_TO_MUJOCO @ root_rotations[frame]
                         @ SOURCE_TO_MUJOCO.T)
        base[3:7] = quat.from_xform(root_rotation)
        target_positions = (SOURCE_TO_MUJOCO
                            @ posed[frame, PICK2_IDS, :, None]).squeeze(-1)

        def residual(joint_angles):
            data.qpos[:] = base
            data.qpos[7:36] = joint_angles
            mujoco.mj_kinematics(model, data)
            # Tiny temporal regularization removes otherwise-unobservable
            # twist ambiguity without materially affecting joint positions.
            return np.concatenate([
                (data.xpos[BODY_IDS] - target_positions).ravel(),
                1e-3 * (joint_angles - previous),
            ])

        result = least_squares(residual, previous, bounds=(lower, upper),
                               max_nfev=80, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        position_residual = residual(result.x)[:3 * len(BODY_IDS)]
        rms = float(np.sqrt(np.mean(position_residual ** 2)))
        if not result.success or rms > MAX_RMS_ERROR:
            raise RuntimeError(
                f"pick2 retarget failed at frame {frame}: "
                f"success={result.success}, RMS={rms:.6f} m")
        base[7:36] = result.x
        qpos[frame] = base
        previous = result.x
        rms_errors.append(rms)

    qpos[:, 3:7] = quat.unroll(qpos[:, 3:7])
    phase, contact = _phase_and_contact(len(qpos))
    return qpos, phase, contact, float(max(rms_errors))


def retarget_file(source_path, output_path):
    """Retarget and save the standard qpos/contact representation."""
    qpos, phase, contact, max_rms = retarget(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, qpos=qpos, phase=phase, contact=contact,
                        fps=np.array(FPS), retarget_max_rms=np.array(max_rms))
    return max_rms

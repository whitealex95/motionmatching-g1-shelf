"""Turn a kimodo pick-up motion (.npz) into the pick clip + scene meta.

Steps:
  1. Convert the kimodo skeleton motion to G1 qpos (needs the kimodo package).
  2. Find where the right palm stops while reaching out: that is the grab.
  3. Put the vase at the palm stop, and size a shelf under it.
  4. Label phases (idle / reach / grasp / lift / hold) and the contact flag.

Writes data/g1_shelf/pick.npz and data/g1_shelf/meta.json.

    python make_clip.py ~/Projects/kimodo/output_shelf.npz
"""
import argparse
import json
import os

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G1_XML = os.path.join(ROOT, "assets", "unitree_g1", "g1.xml")
OUT_DIR = os.path.join(ROOT, "data", "g1_shelf")

FPS = 30.0
PALM_OFFSET = np.array([0.10, -0.007, 0.0])   # palm point in the wrist frame

SPEED_MOVE = 0.25       # palm speed above this = the arm is moving (m/s)
SPEED_STOP = 0.15       # palm speed below this = the arm stopped
EXT_MIN = 0.30          # palm this far from the root = reaching out (m)

GRIP_AHEAD = 0.04       # vase neck sits this far past the palm point
GRASP_H = 0.16          # grasp height above the vase base (vase neck)
SHELF_SETBACK = 0.13    # shelf front edge, this far behind the vase
SHELF_DEPTH = 0.32
SHELF_WIDTH = 0.90

PHASES = ["idle", "reach", "grasp", "lift", "hold"]
IDLE, REACH, GRASP, LIFT, HOLD = range(5)

JOINT_NAMES = None      # filled from the model


def kimodo_to_qpos(npz_path):
    """Kimodo motion npz -> (T, 36) G1 qpos at 30 fps."""
    from kimodo.exports.motion_io import load_kimodo_npz_as_torch
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton.registry import build_skeleton

    motion, num_joints = load_kimodo_npz_as_torch(npz_path, source_fps=FPS)
    skeleton = build_skeleton(num_joints)
    qpos = MujocoQposConverter(skeleton, xml_path=G1_XML).dict_to_qpos(
        motion, numpy=True)
    qpos = np.asarray(qpos)
    if qpos.ndim == 3:
        qpos = qpos[0]
    return qpos.astype(np.float64)


def palm_track(qpos):
    """Right palm world position and pointing axis for every frame."""
    global JOINT_NAMES
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    JOINT_NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(1, model.njnt)]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            "right_wrist_yaw_link")
    T = len(qpos)
    pos = np.zeros((T, 3))
    xaxis = np.zeros((T, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        rot = data.xmat[bid].reshape(3, 3)
        pos[t] = data.xpos[bid] + rot @ PALM_OFFSET
        xaxis[t] = rot[:, 0]
    return pos, xaxis


def find_phases(qpos, palm):
    """Phase label per frame, from the palm speed and how far it reaches."""
    T = len(palm)
    speed = np.zeros(T)
    speed[1:] = np.linalg.norm(np.diff(palm, axis=0), axis=1) * FPS
    ext = np.linalg.norm(palm[:, :2] - qpos[:, :2], axis=1)

    move = int(np.argmax(speed > SPEED_MOVE))
    stopped = (speed < SPEED_STOP) & (ext > EXT_MIN)
    stopped[:move + 5] = False
    grasp = int(np.argmax(stopped))
    if not stopped[grasp]:
        raise RuntimeError("no frame where the reaching hand stops")
    lift = grasp + int(np.argmax(speed[grasp:] > SPEED_MOVE))
    fast = np.where(speed > 0.2)[0]
    hold = int(fast[-1]) + 1 if len(fast) else T - 1

    phase = np.full(T, IDLE, np.int32)
    phase[move:grasp] = REACH
    phase[grasp:lift] = GRASP
    phase[lift:hold] = LIFT
    phase[hold:] = HOLD
    return phase, grasp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="kimodo motion .npz")
    args = ap.parse_args()

    qpos = kimodo_to_qpos(args.npz)
    palm, xaxis = palm_track(qpos)
    phase, grasp = find_phases(qpos, palm)
    contact = phase >= GRASP

    # The vase neck goes a bit past the palm point, along where the hand points.
    ahead = xaxis[grasp] * np.array([1.0, 1.0, 0.0])
    ahead /= np.linalg.norm(ahead) + 1e-9
    neck = palm[grasp] + GRIP_AHEAD * ahead
    vase_pos = [float(neck[0]), float(neck[1]), float(neck[2] - GRASP_H)]

    meta = dict(
        fps=FPS,
        qpos="root xyz, root quat wxyz, 29 joint angles (MuJoCo order)",
        phases=PHASES,
        joint_names=JOINT_NAMES,
        shelf=dict(
            vase_pos=vase_pos,
            grasp_frame=int(grasp),
            palm_offset=PALM_OFFSET.tolist(),
            shelf_top_z=vase_pos[2],
            shelf_front_x=vase_pos[0] - SHELF_SETBACK,
            shelf_center_y=vase_pos[1],
            shelf_depth=SHELF_DEPTH,
            shelf_width=SHELF_WIDTH,
        ),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "pick.npz"),
             qpos=qpos, phase=phase, contact=contact, fps=FPS)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    counts = {PHASES[p]: int((phase == p).sum()) for p in range(len(PHASES))}
    print(f"{len(qpos)} frames -> {OUT_DIR}/pick.npz")
    print(f"phases: {counts}")
    print(f"grab at frame {grasp}, vase at {np.round(vase_pos, 3).tolist()}")


if __name__ == "__main__":
    main()

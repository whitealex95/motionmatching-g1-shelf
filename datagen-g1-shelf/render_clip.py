"""Render the pick clip: the G1 grabs the vase from the shelf.

The robot plays the recorded qpos. The vase stands on the shelf until the
contact flag turns on; from then on it keeps a fixed pose relative to the
palm, so it moves with the hand. No physics anywhere.

    MUJOCO_GL=egl python render_clip.py [-o out/pick.mp4]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import shelf_model as SM

DATA_DIR = os.path.join(ROOT, "data", "g1_shelf")
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])


def quat_inv(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rot(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_inv(q))[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join("out", "pick.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import imageio

    clip = np.load(os.path.join(DATA_DIR, "pick.npz"))
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    qpos, contact = clip["qpos"], clip["contact"]
    fps = float(clip["fps"])

    model, ids = SM.build_model(SCENE_XML, meta["shelf"],
                                off_w=args.width, off_h=args.height)
    data = mujoco.MjData(model)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              "right_wrist_yaw_link")

    vase_rest = data.mocap_pos[ids["vase_mocap"]].copy()
    rest_quat = np.array([1.0, 0.0, 0.0, 0.0])
    rel_pos = rel_quat = None

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.azimuth, cam.elevation = 2.6, 155.0, -18.0
    cam.lookat[:] = [0.4, 0.0, 0.7]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = imageio.get_writer(out, fps=int(fps), codec="libx264",
                                quality=8, macro_block_size=None)
    for t in range(len(qpos)):
        data.qpos[0:36] = qpos[t]
        mujoco.mj_kinematics(model, data)
        palm_p = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ np.array(
            meta["shelf"]["palm_offset"])
        palm_q = data.xquat[wrist].copy()

        if contact[t]:
            if rel_pos is None:     # first contact frame: lock the offset
                rel_pos = quat_rot(quat_inv(palm_q), vase_rest - palm_p)
                rel_quat = quat_mul(quat_inv(palm_q), rest_quat)
                SM.set_vase_color(model, ids, held=True)
            data.mocap_pos[ids["vase_mocap"]] = palm_p + quat_rot(palm_q, rel_pos)
            data.mocap_quat[ids["vase_mocap"]] = quat_mul(palm_q, rel_quat)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    print(f"wrote {len(qpos)} frames -> {out}")


if __name__ == "__main__":
    main()

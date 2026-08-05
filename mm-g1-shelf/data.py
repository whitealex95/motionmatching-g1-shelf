"""Build / load the motion library: LAFAN locomotion + the pick clip.

All clips are concatenated into one array, with per-frame heading, foot
positions and right-palm pose precomputed. Cached to data/motion_lib.npz.

The pick clip (data/g1_shelf/pick.npz, already 30 Hz) is shifted by
SHELF_ORIGIN so the shelf sits ahead of the robot spawn; its reach..hold
span is labeled SKILL_PICK.
"""
import os
import json
import pickle
import numpy as np

import config as C
from g1_model import G1Model, csv_to_qpos, quat_wxyz_yaw


def _gmr_rows(name, data_dir):
    with open(os.path.join(data_dir, name + ".pkl"), "rb") as f:
        d = pickle.load(f)
    return np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1)


def _load_loco_clip(name, data_dir=C.DATA_DIR):
    rows = _gmr_rows(name, data_dir)
    s, e = C.CLIP_TRIM.get(name, (0, len(rows)))
    return csv_to_qpos(rows[s:min(e, len(rows))])


def _load_pick_clip():
    d = np.load(os.path.join(C.SHELF_DATA_DIR, "pick.npz"))
    qpos = d["qpos"].astype(np.float64)
    qpos[:, 0:2] += np.array(C.SHELF_ORIGIN)
    return qpos, d["phase"].astype(np.int32), d["contact"].astype(np.float64)


def build_library(out=C.LIB_PATH):
    model = G1Model()
    with open(C.SHELF_META) as f:
        meta = json.load(f)["shelf"]

    # Locomotion clips are added twice (normal + mirrored). The pick clip is
    # not mirrored: the grab is right-handed.
    loaded = []          # (name, qpos, skill, contact, phase)
    for name in C.CLIPS:
        if not os.path.exists(os.path.join(C.DATA_DIR, name + ".pkl")):
            continue
        q = _load_loco_clip(name)
        zero = np.zeros(len(q))
        none = np.full(len(q), -1, np.int32)
        loco = np.zeros(len(q), np.int32)
        loaded.append((name, q, loco, zero, none))
        if C.MIRROR:
            loaded.append((name + "_mirror", model.mirror_qpos(q), loco, zero, none))

    q, phase, contact = _load_pick_clip()
    skill = np.where(phase >= C.PHASE_REACH, C.SKILL_PICK, C.SKILL_LOCO)
    loaded.append(("pick", q, skill.astype(np.int32), contact, phase))

    qpos, skill, contact, phase = [], [], [], []
    clip_id, frame_in_clip, lengths, names = [], [], [], []
    for cid, (name, qc, sk, ct, ph) in enumerate(loaded):
        n = len(qc)
        qpos.append(qc); skill.append(sk); contact.append(ct); phase.append(ph)
        clip_id.append(np.full(n, cid)); frame_in_clip.append(np.arange(n))
        lengths.append(n); names.append(name)
        tag = ", pick" if sk.any() else ""
        print(f"  [{cid}] {name}: {n} frames{tag}")

    qpos = np.concatenate(qpos)
    feet, hand_pos, hand_dir = model.fk(qpos)
    yaw = quat_wxyz_yaw(qpos[:, 3:7])
    tree = model.body_tree()

    vase = np.array(meta["vase_pos"], np.float64)
    vase[0:2] += np.array(C.SHELF_ORIGIN)

    np.savez_compressed(
        out,
        qpos=qpos.astype(np.float32),
        feet_world=feet.astype(np.float32),
        hand_pos=hand_pos.astype(np.float32),
        hand_dir=hand_dir.astype(np.float32),
        yaw=yaw.astype(np.float32),
        clip_id=np.concatenate(clip_id).astype(np.int32),
        frame_in_clip=np.concatenate(frame_in_clip).astype(np.int32),
        lengths=np.array(lengths, np.int32),
        clip_names=np.array(names),
        skill=np.concatenate(skill).astype(np.int32),
        contact=np.concatenate(contact).astype(np.float32),
        phase=np.concatenate(phase).astype(np.int32),
        vase_pos=vase,                       # world, base of the vase
        shelf_dir=np.array([-1.0, 0.0]),     # the way the shelf faces
        lib_version=np.array(C.LIB_VERSION),
        **tree,
    )
    print(f"Saved library: {qpos.shape[0]} frames, {len(loaded)} clips -> {out}")
    return out


def load_library(path=C.LIB_PATH):
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        version = int(d["lib_version"]) if "lib_version" in d.files else 0
        if version != C.LIB_VERSION:
            print(f"Cache is stale (library v{version} != v{C.LIB_VERSION}); "
                  "rebuilding...")
            os.remove(path)
    if not os.path.exists(path):
        build_library(out=path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_library()

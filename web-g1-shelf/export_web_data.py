"""Export everything the web demo needs into web-g1-shelf/data/.

  model.json   kinematic tree (per body: parent, local pos/quat, hinge axis)
  mesh.json/.bin  the G1 visual meshes, vertices pre-moved into body frames
  shelf.json   world placement of the IKEA GLBs (also copied into data/)
  mm.json/.bin the motion database + every constant the JS matcher needs

Run from the repo root with the mujoco env:

    python web-g1-shelf/export_web_data.py
"""
import json
import os
import shutil
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mm-g1-shelf"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import config as C
import quat
import shelf_model as SM
from data import load_library
from controller import MotionMatcher

OUT = os.path.join(ROOT, "web-g1-shelf", "data")


def export_model(m):
    bodies = []
    for b in range(1, m.nbody):
        axis, qadr = None, -1
        for j in range(m.njnt):
            if (m.jnt_bodyid[j] == b
                    and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE):
                axis, qadr = m.jnt_axis[j].tolist(), int(m.jnt_qposadr[j])
                break
        bodies.append(dict(
            name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
            parent=int(m.body_parentid[b]) - 1,
            pos=m.body_pos[b].tolist(),
            quat=m.body_quat[b].tolist(),
            axis=axis, qadr=qadr))
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(dict(bodies=bodies, nbody=len(bodies)), f)
    return bodies


def verify_fk(m, bodies):
    """The JS FK formula must match MuJoCo exactly."""
    data = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        qpos = np.zeros(m.nq)
        qpos[0:3] = rng.uniform(-1, 1, 3)
        q = rng.normal(size=4); qpos[3:7] = q / np.linalg.norm(q)
        qpos[7:] = rng.uniform(-0.5, 0.5, m.nq - 7)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(m, data)
        wp = [None] * len(bodies)
        wq = [None] * len(bodies)
        for i, b in enumerate(bodies):
            if b["parent"] < 0:
                wp[i], wq[i] = qpos[0:3], qpos[3:7]
            else:
                p, r = wp[b["parent"]], wq[b["parent"]]
                wp[i] = p + quat.mul_vec(r, np.array(b["pos"]))
                r = quat.mul(r, np.array(b["quat"]))
                if b["qadr"] >= 0:
                    r = quat.mul(r, quat.from_angle_axis(
                        np.float64(qpos[b["qadr"]]), np.array(b["axis"])))
                wq[i] = r
            worst = max(worst, float(np.linalg.norm(wp[i] - data.xpos[i + 1])))
    assert worst < 1e-6, worst
    print(f"FK check ok (worst {worst:.2e} m)")


def export_meshes(m):
    verts, faces, geoms = [], [], []
    nv = ni = 0
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va = m.mesh_vertadr[mid]; vn = m.mesh_vertnum[mid]
        fa = m.mesh_faceadr[mid]; fn = m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        vb = m.geom_pos[g] + quat.mul_vec(m.geom_quat[g], v)
        assert vn < 65536              # uint16 indices, local to each geom
        rgba = (m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0
                else m.geom_rgba[g])
        geoms.append(dict(body=int(m.geom_bodyid[g]) - 1,
                          vstart=nv, vcount=int(vn),
                          istart=ni, icount=int(fn) * 3,
                          rgba=[float(c) for c in rgba[:3]]))
        verts.append(vb.astype(np.float32))
        faces.append(m.mesh_face[fa:fa + fn].astype(np.uint16).ravel())
        nv += int(vn); ni += int(fn) * 3
    pos = np.concatenate(verts)
    idx = np.concatenate(faces)
    blob = pos.tobytes() + idx.tobytes()
    with open(os.path.join(OUT, "mesh.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(OUT, "mesh.json"), "w") as f:
        json.dump(dict(nverts=nv, nidx=ni, idx_byte_offset=pos.nbytes,
                       geoms=geoms), f)
    print(f"meshes: {len(geoms)} geoms, {nv} verts, {ni // 3} tris")


def export_shelf():
    """Copy the IKEA GLBs and write their world placement (same math as
    tools/shelf_model.py: the BILLY body origin sits below the bottle target
    by the shelf-board height, yawed -90 deg to face the robot spawn)."""
    with open(C.SHELF_META) as f:
        meta = json.load(f)["shelf"]
    ox, oy = C.SHELF_ORIGIN
    vx, vy, vz = meta["vase_pos"]
    vx += ox
    vy += oy

    shutil.copyfile(SM.BILLY_GLB, os.path.join(OUT, "billy.glb"))
    shutil.copyfile(SM.BOTTLE_GLB, os.path.join(OUT, "bottle.glb"))

    s = float(np.sqrt(0.5))
    billy = dict(glb="billy.glb",
                 pos=[vx, vy, vz - SM.BILLY_BOTTLE_SHELF_Z],
                 quat=[s, 0.0, 0.0, -s])
    vase = dict(glb="bottle.glb", rest=[vx, vy, vz])
    with open(os.path.join(OUT, "shelf.json"), "w") as f:
        json.dump(dict(billy=billy, vase=vase), f)
    print(f"shelf: billy at {billy['pos']}, bottle at {vase['rest']}")


def export_mm(lib):
    m = MotionMatcher(lib)
    db = m.db

    search_clips = []
    for ci, (rs, re) in enumerate(zip(db["starts"], db["stops"])):
        if not lib["skill"][rs:re].any() and re - rs > C.HORIZONS[-1]:
            search_clips.append(ci)

    arrays = dict(
        Xloco=db["dbs"]["loco"]["X"].astype(np.float32),
        locoOffset=db["dbs"]["loco"]["offset"].astype(np.float32),
        locoScale=db["dbs"]["loco"]["scale"].astype(np.float32),
        rawXpos=db["rawXpos"].astype(np.float32),
        rawXvel=db["rawXvel"].astype(np.float32),
        dof=db["dof"].astype(np.float32),
        dofVel=db["dofVel"].astype(np.float32),
        simPos=db["simPos"].astype(np.float32),
        simTheta=db["simTheta"].astype(np.float32),
        simVel=db["simVel"].astype(np.float32),
        yawRate=db["yawRate"].astype(np.float32),
        pelvLocalPos=db["pelvLocalPos"].astype(np.float32),
        pelvLocalVel=db["pelvLocalVel"].astype(np.float32),
        pelvLocalRot=db["pelvLocalRot"].astype(np.float32),
        pelvLocalAng=db["pelvLocalAng"].astype(np.float32),
        starts=db["starts"].astype(np.int32),
        stops=db["stops"].astype(np.int32),
        skill=lib["skill"].astype(np.int32),
        contact=lib["contact"].astype(np.float32),
        phase=lib["phase"].astype(np.int32),
        search_clips=np.array(search_clips, np.int32),
    )

    header, blob, off = {}, b"", 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        header[name] = dict(dtype=str(arr.dtype), shape=list(arr.shape),
                            offset=off)
        blob += arr.tobytes()
        off += arr.nbytes

    meta = dict(
        fps=C.FPS, ndof=29, horizons=C.HORIZONS,
        max_speed=C.MAX_SPEED, walk_scale=C.WALK_SCALE,
        search_time=C.SEARCH_TIME, current_bias=C.CURRENT_BIAS,
        inert_halflife=C.INERT_HALFLIFE,
        vel_halflife=C.VEL_HALFLIFE, rot_halflife=C.ROT_HALFLIFE,
        skill_loco=C.SKILL_LOCO, skill_pick=C.SKILL_PICK,
        phase_idle=C.PHASE_IDLE,
        move_overshoot=C.MOVE_OVERSHOOT,
        move_arrive_near=C.MOVE_ARRIVE_NEAR,
        move_arrive_yaw=C.MOVE_ARRIVE_YAW,
        move_timeout=C.MOVE_TIMEOUT,
        snap_radius=C.SNAP_RADIUS, snap_halflife=C.SNAP_HALFLIFE,
        palm_offset=C.PALM_OFFSET, arm_qpos_start=C.ARM_QPOS.start,
        # Precomputed by the Python controller so both sides agree exactly.
        pick_entry=int(m.pick_entry), pick_lo=int(m.pick_lo),
        pick_hi=int(m.pick_hi),
        stance_xy=m.stance_xy.tolist(), stance_yaw=float(m.stance_yaw),
        route_wp=m.route_wp.tolist(),
        vase_rest=m.vase_rest.tolist(),
        snap_pos=m.snap_pos.tolist(), snap_quat=m.snap_quat.tolist(),
        clip_names=[str(n) for n in lib["clip_names"]],
        n_frames=int(len(lib["qpos"])),
        arrays=header)

    with open(os.path.join(OUT, "mm.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(OUT, "mm.json"), "w") as f:
        json.dump(meta, f)
    print(f"mm: {len(arrays)} arrays, {off / 1e6:.1f} MB, "
          f"{len(search_clips)} loco clips searchable")


def main():
    os.makedirs(OUT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    bodies = export_model(model)
    verify_fk(model, bodies)
    export_meshes(model)
    export_shelf()
    export_mm(load_library())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Feature database with a smoothed simulation root.

The locomotion feature (27-D) is the GenoView one:
  Xpos (6)      local foot positions relative to the sim root
  Xvel (9)      local velocities of both feet + the pelvis
  XtrajPos (6)  future sim-root xy at +10/+20/+30 frames
  XtrajDir (6)  future sim heading xy at the same horizons

The pick database swaps the trajectory for the interaction blocks:
  handPos / handVel / handDir (3+3+3)   right palm pose (heading frame)
  vasePos (2) / shelfDir (2)            where the vase is and which way the
                                        shelf faces, in the heading frame
  robotPos (2) / robotDir (2)           robot stance in the static shelf frame
  held (1)                              vase-in-hand flag {0,1}
"""
import numpy as np
from scipy.signal import savgol_filter

import config as C
import quat

FPS = C.FPS
HORIZONS = np.array(C.HORIZONS)
FORWARD = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def yaw_quat(theta):
    """Quaternion (wxyz) for a rotation of theta about world +Z."""
    return quat.from_angle_axis(np.asarray(theta), UP)


def heading_dir(rootquat):
    """World forward direction (xy, z=0, normalized) of a root quaternion."""
    fwd = quat.mul_vec(rootquat, FORWARD) * np.array([1.0, 1.0, 0.0])
    return fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)


def smooth_root(pelvisPos_world, headDir):
    """Smoothed simulation root for one clip."""
    n = len(pelvisPos_world)
    pw = min(C.ROOT_POS_SMOOTH, n if n % 2 == 1 else n - 1)
    dw = min(C.ROOT_DIR_SMOOTH, n if n % 2 == 1 else n - 1)
    simXY = pelvisPos_world[:, :2]
    if pw >= 5:
        simXY = savgol_filter(simXY, pw, 3, axis=0, mode='interp')
    simPos = np.concatenate([simXY, np.zeros((n, 1))], axis=1)
    d = headDir[:, :2].copy()
    if dw >= 5:
        d = savgol_filter(d, dw, 3, axis=0, mode='interp')
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    headDirSmooth = np.concatenate([d, np.zeros((n, 1))], axis=1)
    simTheta = np.arctan2(d[:, 1], d[:, 0])
    return simPos, simTheta, headDirSmooth


def central_diff(x, fps):
    v = np.empty_like(x)
    if len(x) < 4:
        v[:] = (np.gradient(x, axis=0) * fps) if len(x) > 1 else 0.0
        return v
    v[1:-1] = 0.5 * (x[2:] - x[1:-1]) * fps + 0.5 * (x[1:-1] - x[:-2]) * fps
    v[0] = v[1] - (v[3] - v[2])
    v[-1] = v[-2] + (v[-2] - v[-3])
    return v


def central_diff_ang(rot, fps):
    n = len(rot)
    ang = np.zeros((n, 3))
    if n < 4:
        if n >= 2:
            ang[1:] = quat.to_scaled_angle_axis(
                quat.abs(quat.mul_inv(rot[1:], rot[:-1]))) * fps
            ang[0] = ang[1]
        return ang
    ang[1:-1] = (0.5 * quat.to_scaled_angle_axis(
                     quat.abs(quat.mul_inv(rot[2:], rot[1:-1]))) * fps +
                 0.5 * quat.to_scaled_angle_axis(
                     quat.abs(quat.mul_inv(rot[1:-1], rot[:-2]))) * fps)
    ang[0] = ang[1] - (ang[3] - ang[2])
    ang[-1] = ang[-2] + (ang[-2] - ang[-3])
    return ang


def shelf_local_blocks(qh, rootPos, vase_pos, shelf_dir, held):
    """The live shelf feature blocks, batched (T,...) or single-frame.
    Returns (vasePos (..2), shelfDir (..2), robotPos (..2), robotDir (..2),
    held (..1))."""
    vase3 = np.zeros(np.shape(rootPos))
    vase3[..., 0:2] = vase_pos[0:2]
    dir3 = np.zeros(np.shape(rootPos))
    dir3[..., 0:2] = shelf_dir
    vpos = quat.inv_mul_vec(qh, vase3 - rootPos * np.array([1.0, 1.0, 0.0]))[..., 0:2]
    vdir = quat.inv_mul_vec(qh, dir3)[..., 0:2]
    ax = np.asarray(shelf_dir, np.float64)
    ax = ax / np.linalg.norm(ax)
    ay = np.array([-ax[1], ax[0]])
    rel = np.asarray(rootPos)[..., 0:2] - vase_pos[0:2]
    rpos = np.stack([rel @ ax, rel @ ay], -1)
    head = quat.mul_vec(qh, FORWARD)[..., 0:2]
    rdir = np.stack([head @ ax, head @ ay], -1)
    held = np.reshape(held, np.shape(vpos)[:-1] + (1,)).astype(float)
    return vpos, vdir, rpos, rdir, held


def build_db(lib):
    """Assemble the feature DB (a dict) from the library."""
    qpos = lib["qpos"].astype(np.float64)
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(qpos))
    spans = list(zip(starts, stops))

    rootQuat = qpos[:, 3:7].copy()
    dof = qpos[:, 7:].copy()
    footL = lib["feet_world"][:, 0].astype(np.float64)
    footR = lib["feet_world"][:, 1].astype(np.float64)
    pelvis = qpos[:, 0:3]
    headDirRaw = heading_dir(rootQuat)
    handPosW = lib["hand_pos"].astype(np.float64)
    handDirW = lib["hand_dir"].astype(np.float64)

    T = len(qpos)
    simPos = np.zeros((T, 3)); simTheta = np.zeros(T); headDir = np.zeros((T, 3))
    pelvLocalPos = np.zeros((T, 3)); pelvLocalRot = np.zeros((T, 4))
    for rs, re in spans:
        sp, st, hd = smooth_root(pelvis[rs:re], headDirRaw[rs:re])
        simPos[rs:re], simTheta[rs:re], headDir[rs:re] = sp, st, hd
        qh = yaw_quat(st)
        pelvLocalPos[rs:re] = quat.inv_mul_vec(qh, pelvis[rs:re] - sp)
        pelvLocalRot[rs:re] = quat.mul(quat.inv(qh), rootQuat[rs:re])

    def clipwise_vel(arr):
        v = np.zeros_like(arr)
        for rs, re in spans:
            v[rs:re] = central_diff(arr[rs:re], FPS)
        return v

    footLvel, footRvel = clipwise_vel(footL), clipwise_vel(footR)
    pelvisVel, simVel = clipwise_vel(pelvis), clipwise_vel(simPos)
    dofVel, pelvLocalVel = clipwise_vel(dof), clipwise_vel(pelvLocalPos)
    handVelW = clipwise_vel(handPosW)
    yawRate = np.zeros(T)
    pelvLocalAng = np.zeros((T, 3))
    for rs, re in spans:
        yawRate[rs:re] = central_diff(np.unwrap(simTheta[rs:re])[:, None], FPS)[:, 0]
        pelvLocalAng[rs:re] = central_diff_ang(pelvLocalRot[rs:re], FPS)

    qh_all = yaw_quat(simTheta)
    to_local = lambda v: quat.inv_mul_vec(qh_all, v)

    Xpos = np.concatenate([to_local(footL - simPos), to_local(footR - simPos)], -1)
    Xvel = np.concatenate([to_local(footLvel), to_local(footRvel),
                           to_local(pelvisVel)], -1)
    XtrajPos = np.zeros((T, 6))
    XtrajDir = np.zeros((T, 6))
    for rs, re in spans:
        idx = np.arange(rs, re)
        for k, h in enumerate(HORIZONS):
            ft = np.clip(idx + h, rs, re - 1)
            XtrajPos[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(
                qh_all[rs:re], simPos[ft] - simPos[rs:re])[:, 0:2]
            XtrajDir[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(
                qh_all[rs:re], headDir[ft])[:, 0:2]

    handPos = to_local(handPosW - simPos)
    handVel = to_local(handVelW)
    handDir = to_local(handDirW)
    vasePos, shelfDir, robotPos, robotDir, held = shelf_local_blocks(
        qh_all, simPos, lib["vase_pos"].astype(np.float64),
        lib["shelf_dir"].astype(np.float64),
        lib["contact"].astype(np.float64))

    skill = lib["skill"]
    masks = {"loco": skill == C.SKILL_LOCO, "pick": skill == C.SKILL_PICK}

    def make_db(blocks, mask):
        if not mask.any():
            mask = np.ones(T, bool)
        X = np.concatenate([b for b, _ in blocks], -1)
        offset = X[mask].mean(0)
        scale = np.concatenate([np.repeat(b[mask].std(0).mean() / w, b.shape[1])
                                for b, w in blocks])
        scale = np.where(scale < 1e-5, 1.0, scale)
        return ((X - offset) / scale).astype(np.float32), offset, scale

    pose = [(Xpos, 1.0), (Xvel, 1.0)]
    traj = [(XtrajPos, 1.0), (XtrajDir, 1.0)]
    pick = [(handPos, C.HAND_POS_WEIGHT), (handVel, C.HAND_VEL_WEIGHT),
            (handDir, C.HAND_DIR_WEIGHT), (vasePos, C.VASE_POS_WEIGHT),
            (shelfDir, C.SHELF_DIR_WEIGHT),
            (robotPos, C.ROBOT_POS_WEIGHT), (robotDir, C.ROBOT_DIR_WEIGHT),
            (held, C.HELD_WEIGHT)]
    dbs = {
        "loco": make_db(pose + traj, masks["loco"]),     # 27-D
        "pick": make_db(pose + pick, masks["pick"]),     # 15 + 18 = 33-D
    }
    dbs = {k: dict(X=Xn, offset=off, scale=sc) for k, (Xn, off, sc) in dbs.items()}

    return dict(
        starts=starts, stops=stops, spans=spans,
        dof=dof, dofVel=dofVel,
        simPos=simPos, simTheta=simTheta, simVel=simVel, yawRate=yawRate,
        pelvLocalPos=pelvLocalPos, pelvLocalVel=pelvLocalVel,
        pelvLocalRot=pelvLocalRot, pelvLocalAng=pelvLocalAng,
        rawXpos=Xpos, rawXvel=Xvel, rawTrajPos=XtrajPos, rawTrajDir=XtrajDir,
        rawHandPos=handPos, rawHandVel=handVel, rawHandDir=handDir,
        dbs=dbs)

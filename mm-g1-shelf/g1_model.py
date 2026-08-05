"""MuJoCo G1 wrapper: qpos conversion + forward kinematics for features."""
import numpy as np
import mujoco

import config as C


def csv_to_qpos(rows):
    """Dataset rows [...,quat_xyzw,...] -> MuJoCo qpos [...,quat_wxyz,...]."""
    rows = np.atleast_2d(rows).astype(np.float64)
    q = rows.copy()
    q[:, 3:7] = rows[:, [6, 3, 4, 5]]
    return q


class G1Model:
    """Loads the menagerie G1 and exposes batched FK for feature extraction."""

    def __init__(self, xml=C.SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in C.FOOT_BODIES
        ]
        self._build_mirror_map()

    def fk(self, qpos_seq):
        """One FK pass per frame. Returns world foot positions (T,2,3)."""
        qpos_seq = np.atleast_2d(qpos_seq)
        T = len(qpos_seq)
        feet = np.empty((T, len(self.foot_ids), 3))
        for t, q in enumerate(qpos_seq):
            self.data.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.data)
            for k, bid in enumerate(self.foot_ids):
                feet[t, k] = self.data.xpos[bid]
        return feet

    def body_tree(self):
        """Kinematic tree arrays for the arm IK: per body (world excluded,
        pelvis first) the parent index, local pos/quat, hinge axis and qpos
        address; plus the right-arm joint limits for qpos 29..35."""
        m = self.model
        parent, pos, qt, axis, qadr = [], [], [], [], []
        for b in range(1, m.nbody):
            ax, qa = np.zeros(3), -1
            for j in range(m.njnt):
                if (m.jnt_bodyid[j] == b
                        and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE):
                    ax, qa = m.jnt_axis[j].copy(), int(m.jnt_qposadr[j])
                    break
            parent.append(int(m.body_parentid[b]) - 1)
            pos.append(m.body_pos[b].copy())
            qt.append(m.body_quat[b].copy())
            axis.append(ax)
            qadr.append(qa)
        lo, hi = np.zeros(7), np.zeros(7)
        for j in range(m.njnt):
            a = int(m.jnt_qposadr[j])
            if C.ARM_QPOS.start <= a < C.ARM_QPOS.stop:
                lo[a - C.ARM_QPOS.start], hi[a - C.ARM_QPOS.start] = m.jnt_range[j]
        return dict(
            body_parent=np.array(parent, np.int32),
            body_pos=np.array(pos, np.float64),
            body_quat=np.array(qt, np.float64),
            body_axis=np.array(axis, np.float64),
            body_qadr=np.array(qadr, np.int32),
            arm_lo=lo, arm_hi=hi)

    # --- sagittal mirror (left/right reflection) ---------------------------
    def _build_mirror_map(self):
        m = self.model
        self._mir_src, self._mir_dst, self._mir_sign = [], [], []
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            twin = (name.replace("left", "right") if "left" in name
                    else name.replace("right", "left") if "right" in name else name)
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, twin)
            sign = 1.0 if abs(m.jnt_axis[j][1]) > 0.5 else -1.0
            self._mir_src.append(m.jnt_qposadr[j])
            self._mir_dst.append(m.jnt_qposadr[tid])
            self._mir_sign.append(sign)
        self._mir_src = np.array(self._mir_src)
        self._mir_dst = np.array(self._mir_dst)
        self._mir_sign = np.array(self._mir_sign)

    def mirror_qpos(self, qpos_seq):
        """Left/right-mirror a (T,36) qpos sequence (wxyz)."""
        q = np.atleast_2d(qpos_seq).astype(np.float64)
        out = q.copy()
        out[:, 1] = -q[:, 1]
        out[:, 4] = -q[:, 4]; out[:, 6] = -q[:, 6]
        out[:, self._mir_dst] = self._mir_sign * q[:, self._mir_src]
        return out

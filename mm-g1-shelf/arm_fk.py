"""Forward kinematics for the right palm, from the body tree stored in the
motion library. Used to keep the vase glued to the hand after the grab."""
import numpy as np

import config as C
import quat

A0 = C.ARM_QPOS.start


class ArmFK:
    def __init__(self, lib):
        self.parent = lib["body_parent"].astype(int)
        self.bpos = lib["body_pos"].astype(np.float64)
        self.bquat = lib["body_quat"].astype(np.float64)
        self.baxis = lib["body_axis"].astype(np.float64)
        self.bqadr = lib["body_qadr"].astype(int)
        self.palm = np.array(C.PALM_OFFSET, np.float64)

        # The 7-body right-arm chain, ordered by qpos address 29..35; its
        # base is the chain root's parent (the torso).
        adr = {int(a): i for i, a in enumerate(self.bqadr)
               if A0 <= a < C.ARM_QPOS.stop}
        self.chain = [adr[a] for a in range(A0, C.ARM_QPOS.stop)]
        self.base = int(self.parent[self.chain[0]])

    def _body_world(self, qpos, body):
        n = len(self.parent)
        wp = np.zeros((n, 3))
        wq = np.zeros((n, 4))
        wp[0] = qpos[0:3]
        wq[0] = qpos[3:7]
        for i in range(1, body + 1):
            p = self.parent[i]
            wp[i] = wp[p] + quat.mul_vec(wq[p], self.bpos[i])
            r = quat.mul(wq[p], self.bquat[i])
            if self.bqadr[i] >= 0:
                r = quat.mul(r, quat.from_angle_axis(
                    np.float64(qpos[self.bqadr[i]]), self.baxis[i]))
            wq[i] = r
        return wp[body], wq[body]

    def palm_pose(self, qpos):
        """World palm position + orientation (wxyz) for a full qpos (36,)."""
        p, r = self._body_world(qpos, self.base)
        q7 = qpos[C.ARM_QPOS].astype(np.float64)
        for k, i in enumerate(self.chain):
            p = p + quat.mul_vec(r, self.bpos[i])
            r = quat.mul(r, self.bquat[i])
            r = quat.mul(r, quat.from_angle_axis(np.float64(q7[k]), self.baxis[i]))
        return p + quat.mul_vec(r, self.palm), r

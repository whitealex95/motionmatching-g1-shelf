"""Right-arm IK overlay: warp the played-back arm so the palm lands on the
vase during the reach (the pick skill plays back relative, so the body
carries a stance offset; the recorded hand trajectory is right in world
coordinates w.r.t. the static shelf).

Damped-least-squares IK over the 7 right-arm joints with finite-difference
Jacobians on the body-tree FK. The solve warm-starts from the previous
frame's correction, keeping it a small change of the recorded motion.
"""
import numpy as np

import config as C
import quat

FORWARD = np.array([1.0, 0.0, 0.0])
A0 = C.ARM_QPOS.start


def _solve6(A, b):
    """Gaussian elimination with partial pivoting for the 6x6 DLS system."""
    A = A.copy()
    b = b.copy()
    n = 6
    for c in range(n):
        p = c + int(np.argmax(np.abs(A[c:, c])))
        if p != c:
            A[[c, p]] = A[[p, c]]
            b[[c, p]] = b[[p, c]]
        inv = 1.0 / A[c, c]
        for r in range(c + 1, n):
            f = A[r, c] * inv
            A[r, c:] -= f * A[c, c:]
            b[r] -= f * b[c]
    x = np.zeros(n)
    for r in range(n - 1, -1, -1):
        x[r] = (b[r] - A[r, r + 1:] @ x[r + 1:]) / A[r, r]
    return x


class ArmIK:
    """Built from the body tree stored in the motion library (data.py)."""

    def __init__(self, lib):
        self.parent = lib["body_parent"].astype(int)
        self.bpos = lib["body_pos"].astype(np.float64)
        self.bquat = lib["body_quat"].astype(np.float64)
        self.baxis = lib["body_axis"].astype(np.float64)
        self.bqadr = lib["body_qadr"].astype(int)
        self.lo = lib["arm_lo"].astype(np.float64)
        self.hi = lib["arm_hi"].astype(np.float64)
        self.palm = np.array(C.PALM_OFFSET, np.float64)

        # The 7-body arm chain, ordered by qpos address 29..35; its base is
        # the chain root's parent (the torso).
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

    def _chain_fk(self, bp, bq, q7):
        p, r = bp, bq
        for k, i in enumerate(self.chain):
            p = p + quat.mul_vec(r, self.bpos[i])
            r = quat.mul(r, self.bquat[i])
            r = quat.mul(r, quat.from_angle_axis(np.float64(q7[k]), self.baxis[i]))
        return p + quat.mul_vec(r, self.palm), r

    def _palm(self, bp, bq, q7):
        """Chain FK: palm position + palm x-axis for arm angles q7."""
        p, r = self._chain_fk(bp, bq, q7)
        return p, quat.mul_vec(r, FORWARD)

    def palm_pose(self, qpos):
        """World palm position + orientation (wxyz) for a full qpos."""
        bp, bq = self._body_world(qpos, self.base)
        return self._chain_fk(bp, bq, qpos[C.ARM_QPOS].astype(np.float64))

    def solve(self, qpos, target_p, target_x, prev_delta):
        """Correction (7,) to add to qpos[29:36] so the palm reaches
        target_p pointing along target_x. Warm-started from prev_delta."""
        bp, bq = self._body_world(qpos, self.base)
        ref = qpos[C.ARM_QPOS].astype(np.float64)
        q = np.clip(ref + prev_delta, self.lo, self.hi)
        w = C.IK_DIR_WEIGHT
        for _ in range(C.IK_ITERS):
            p0, x0 = self._palm(bp, bq, q)
            e = np.concatenate([target_p - p0, w * (target_x - x0)])
            J = np.zeros((6, 7))
            for j in range(7):
                qj = q.copy()
                qj[j] += C.IK_FD_EPS
                p1, x1 = self._palm(bp, bq, qj)
                J[0:3, j] = (p1 - p0) / C.IK_FD_EPS
                J[3:6, j] = w * (x1 - x0) / C.IK_FD_EPS
            A = J @ J.T + (C.IK_LAMBDA * C.IK_LAMBDA) * np.eye(6)
            dq = J.T @ _solve6(A, e)
            dq = np.clip(dq, -C.IK_STEP_CLAMP, C.IK_STEP_CLAMP)
            q = np.clip(q + dq, self.lo, self.hi)
        return q - ref

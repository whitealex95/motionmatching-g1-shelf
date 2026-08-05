"""Real-time motion matching with a vase-pick skill on top of GenoView loco.

The locomotion core is the same as motionmatching-g1-door: a smoothed
"simulation root" the matcher tracks + integrates, per-clip KD-tree search,
inertialized cuts. On top sits the pick skill, driven by B:

    LOCOMOTION --B--> PICK (idle .. reach .. grasp .. lift .. hold) --> LOCOMOTION

Everything plays back relative to the live root. B arms the grab; once the
entry query loss drops below PICK_ENTER_THRESHOLD the matcher enters the
pick clip at the stance-nearest idle frame and the ride plays to the end;
a grab gate at
the idle->reach boundary aborts stances the arm cannot reach from, and the
reach-phase arm IK retargets the palm onto the recorded hand trajectory,
which lands on the vase. When the contact flag turns on, the vase locks to
the palm with a fixed relative pose and follows the hand from then on.
"""
import numpy as np
from scipy.spatial import cKDTree

import config as C
import quat
import shelf
from arm_ik import ArmIK
from features import build_db, shelf_local_blocks, yaw_quat, FORWARD, HORIZONS, FPS
from springs import (DecaySpringDamperPosition, DecaySpringDamperRotation,
                     TrajectorySpringPosition, TrajectorySpringRotation)

DT = C.DT
NDOF = 29
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class MotionMatcher:
    def __init__(self, lib, start_frame=None):
        self.lib = lib
        db = self.db = build_db(lib)
        self.starts, self.stops = db["starts"], db["stops"]
        self.dof, self.dofVel = db["dof"], db["dofVel"]
        self.simPosDB, self.simThetaDB = db["simPos"], db["simTheta"]
        self.simVelDB, self.yawRateDB = db["simVel"], db["yawRate"]
        self.plpDB, self.plvDB = db["pelvLocalPos"], db["pelvLocalVel"]
        self.prDB, self.paDB = db["pelvLocalRot"], db["pelvLocalAng"]
        self.Xloco = db["dbs"]["loco"]["X"]
        self.rawXpos, self.rawXvel = db["rawXpos"], db["rawXvel"]
        self.rawHandPos, self.rawHandVel = db["rawHandPos"], db["rawHandVel"]
        self.rawHandDir = db["rawHandDir"]
        self.clip_id = lib["clip_id"]
        self.skill = lib["skill"]
        self.phase = lib["phase"]
        self.contact = lib["contact"]
        self.vase_rest = lib["vase_pos"].astype(np.float64)   # world, vase base
        self.shelf_dir = lib["shelf_dir"].astype(np.float64)
        self.Ttimes = HORIZONS / FPS
        TAIL = HORIZONS[-1]

        # Locomotion KD-trees: one per pure-locomotion clip. The pick clip is
        # excluded so plain walking never wanders into the interaction.
        self.loco_trees = []
        for rs, re in zip(self.starts, self.stops):
            if self.skill[rs:re].any() or re - rs <= TAIL:
                continue
            self.loco_trees.append((int(rs), int(re),
                                    cKDTree(self.Xloco[rs:re - TAIL])))

        self.pick_enter, self.pick_end_of = shelf.pick_entries(lib)
        self.Xpick = db["dbs"]["pick"]["X"]

        self.armik = ArmIK(lib)
        self.handPosW = lib["hand_pos"].astype(np.float64)
        self.handDirW = lib["hand_dir"].astype(np.float64)
        self.reset(start_frame)

    # --- state --------------------------------------------------------------
    def reset(self, start_frame=None):
        if start_frame is None:
            start_frame = min(self.stops[0] - 1, self.starts[0] + 30)
        self.lo, self.hi = self._clip_bounds(start_frame)
        self.animFrame = int(start_frame)
        self.state = C.SKILL_LOCO
        # Start at the world origin facing +x; the shelf sits ahead at
        # SHELF_ORIGIN, exactly as the shifted clip recorded it.
        self.rootPos = np.zeros(3)
        self.rootVel = np.zeros(3); self.rootAcc = np.zeros(3)
        self.rootAng = np.zeros(3)
        self.rootYaw = 0.0
        self.rootRot = yaw_quat(self.rootYaw)
        self.desiredDir = quat.mul_vec(self.rootRot, FORWARD)
        self.offDof = np.zeros(NDOF); self.offDofVel = np.zeros(NDOF)
        self.offPP = np.zeros(3); self.offPPVel = np.zeros(3)
        self.offPR = IDENTITY.copy(); self.offPAng = np.zeros(3)
        self.searchTimer = 0.0
        self.pick_armed = False
        self.pick_locked = 0
        self.grab_checked = False
        self.pick_err = None
        self.pick_best_entry = -1
        self.ikW = 0.0
        self.ikDelta = np.zeros(7)
        # The vase: on the shelf until the grab, then stuck to the palm.
        self.held = False
        self.vase_pos = self.vase_rest.copy()
        self.vase_quat = IDENTITY.copy()
        self.rel_pos = np.zeros(3)
        self.rel_quat = IDENTITY.copy()
        self.Tpos = np.tile(self.rootPos, (len(HORIZONS), 1))
        self.Tdir = np.tile(self.desiredDir, (len(HORIZONS), 1))

    def _clip_bounds(self, frame):
        r = int(np.searchsorted(self.starts, frame, "right") - 1)
        return int(self.starts[r]), int(self.stops[r])

    @property
    def cur(self):
        return self.animFrame

    @property
    def near_vase(self):
        return (float(np.linalg.norm(self.rootPos[:2] - self.vase_rest[:2]))
                < C.PICK_TRIGGER_RADIUS)

    def state_name(self):
        return {C.SKILL_LOCO: "LOCOMOTION", C.SKILL_PICK: "PICK"}[self.state]

    # --- trigger ------------------------------------------------------------
    def trigger_pick(self):
        """B: arm or disarm the grab. While armed, the pick clip starts by
        itself once the query loss drops below PICK_ENTER_THRESHOLD."""
        if self.pick_locked > 0 or self.state != C.SKILL_LOCO or self.held:
            return
        self.pick_armed = not self.pick_armed

    # --- inertialized cut ---------------------------------------------------
    def _inertialize_into(self, b, lo, hi):
        a = self.animFrame
        self.offDof = (self.offDof + self.dof[a]) - self.dof[b]
        self.offDofVel = (self.offDofVel + self.dofVel[a]) - self.dofVel[b]
        self.offPP = (self.offPP + self.plpDB[a]) - self.plpDB[b]
        self.offPPVel = (self.offPPVel + self.plvDB[a]) - self.plvDB[b]
        self.offPR = quat.abs(quat.mul_inv(
            quat.mul(self.offPR, self.prDB[a]), self.prDB[b]))
        self.offPAng = (self.offPAng + self.paDB[a]) - self.paDB[b]
        self.animFrame, self.lo, self.hi = int(b), int(lo), int(hi)

    # --- pick skill ----------------------------------------------------------
    def _update_pick_err(self):
        """Refresh the pick "query loss": near the vase, the distance to the
        best entry frame; during the skill, the live-vs-playhead distance."""
        self.pick_err = None
        self.pick_best_entry = -1
        if self.state == C.SKILL_PICK:
            Xq = self._query("pick")
            self.pick_err = float(np.linalg.norm(Xq - self.Xpick[self.animFrame]))
        elif (self.state == C.SKILL_LOCO and self.pick_locked == 0
                and self.near_vase and not self.held
                and len(self.pick_enter) > 0):
            Xq = self._query("pick")
            d = np.linalg.norm(self.Xpick[self.pick_enter] - Xq, axis=1)
            k = int(np.argmin(d))
            self.pick_err = float(d[k])
            self.pick_best_entry = int(self.pick_enter[k])

    def _maybe_trigger_pick(self):
        if (not self.pick_armed or self.pick_best_entry < 0
                or self.pick_err > C.PICK_ENTER_THRESHOLD):
            return
        self.pick_armed = False
        entry = self.pick_best_entry
        end = int(self.pick_end_of[entry])
        lo, _ = self._clip_bounds(entry)
        self._inertialize_into(entry, lo, end + 1)
        self.state = C.SKILL_PICK
        self.pick_locked = max(1, end - entry)
        self.grab_checked = False
        self.searchTimer = C.SEARCH_TIME

    def _grab_ok(self):
        """The grab gate: preview the reach/grasp from the current stance
        (future root = live root composed with the recorded root deltas) and
        require every sampled arm IK residual under GRAB_CHECK_TOL."""
        ref = self.animFrame
        span = np.arange(ref, self.hi)
        keep = ((self.phase[span] >= C.PHASE_REACH)
                & (self.phase[span] <= C.PHASE_GRASP))
        span = span[keep]
        if len(span) == 0:
            return False
        idx = np.unique(np.linspace(0, len(span) - 1,
                                    C.GRAB_CHECK_SAMPLES).astype(int))
        dyaw = self.rootYaw - float(self.simThetaDB[ref])
        qd = yaw_quat(dyaw)
        for f in span[idx]:
            rp = self.rootPos + quat.mul_vec(
                qd, self.simPosDB[f] - self.simPosDB[ref])
            ryaw = self.rootYaw + float(self.simThetaDB[f]
                                        - self.simThetaDB[ref])
            qh = yaw_quat(ryaw)
            qpos = np.empty(36)
            qpos[0:3] = rp + quat.mul_vec(qh, self.plpDB[f])
            qpos[3:7] = quat.mul(qh, self.prDB[f])
            qpos[7:] = self.dof[f]
            delta = self.armik.solve(qpos, self.handPosW[f],
                                     self.handDirW[f], np.zeros(7))
            bp, bq = self.armik._body_world(qpos, self.armik.base)
            palm, _ = self.armik._palm(bp, bq, qpos[C.ARM_QPOS] + delta)
            if float(np.linalg.norm(palm - self.handPosW[f])) > C.GRAB_CHECK_TOL:
                return False
        return True

    def _end_skill(self):
        """Leave the PICK skill: fold the applied IK arm correction into the
        inertialization offsets, then hand back to locomotion."""
        self.offDof[C.ARM_DOF] += self.ikW * self.ikDelta
        self.ikW = 0.0
        self.ikDelta = np.zeros(7)
        self.pick_locked = 0
        f, lo, hi = self._search_trees(self.loco_trees, self._query("loco"),
                                       self.Xloco, False)
        self._inertialize_into(f, lo, hi)
        self.state = C.SKILL_LOCO
        self.searchTimer = C.SEARCH_TIME

    # --- desired trajectory (query) -----------------------------------------
    def _predict_trajectory(self, desiredVel, desiredFace):
        desiredVel = np.asarray(desiredVel, float)
        if np.linalg.norm(desiredFace) > 0.01:
            self.desiredDir = (np.asarray(desiredFace, float)
                               / np.linalg.norm(desiredFace))
        elif np.linalg.norm(desiredVel) > 0.01:
            self.desiredDir = desiredVel / np.linalg.norm(desiredVel)
        desiredRot = yaw_quat(np.arctan2(self.desiredDir[1], self.desiredDir[0]))
        dt_col = self.Ttimes[:, None]
        self.Tpos, _, _ = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel,
            C.VEL_HALFLIFE, dt_col)
        Trot, _ = TrajectorySpringRotation(
            self.rootRot, self.rootAng, desiredRot, C.ROT_HALFLIFE, dt_col)
        self.Tdir = quat.mul_vec(Trot, FORWARD)

    def _query(self, name):
        d = self.db["dbs"][name]
        f = self.animFrame
        qh = yaw_quat(self.rootYaw)
        parts = [self.rawXpos[f], self.rawXvel[f]]
        if name == "loco":
            parts.append(quat.inv_mul_vec(qh, self.Tpos - self.rootPos)[:, 0:2].ravel())
            parts.append(quat.inv_mul_vec(qh, self.Tdir)[:, 0:2].ravel())
        else:
            parts.extend([self.rawHandPos[f], self.rawHandVel[f],
                          self.rawHandDir[f]])
            parts.extend(shelf_local_blocks(qh, self.rootPos, self.vase_rest,
                                            self.shelf_dir, float(self.held)))
        q = np.concatenate(parts)
        return (q - d["offset"]) / d["scale"]

    # --- search --------------------------------------------------------------
    def _search_trees(self, trees, Xq, Xself, with_bias, bias_tail=None):
        if bias_tail is None:
            bias_tail = HORIZONS[-1]
        bestF, bestLo, bestHi = self.animFrame, self.lo, self.hi
        if with_bias and self.animFrame < self.hi - bias_tail:
            best = float(np.linalg.norm(Xq - Xself[self.animFrame]) - C.CURRENT_BIAS)
        else:
            best = np.inf
        for rs, re, tree in trees:
            dist, k = tree.query(Xq, eps=C.APPROX_BIAS, distance_upper_bound=best)
            if dist < best:
                best, bestF, bestLo, bestHi = dist, int(rs + k), rs, re
        return bestF, bestLo, bestHi

    def _search_loco(self):
        f, lo, hi = self._search_trees(self.loco_trees, self._query("loco"),
                                       self.Xloco, True)
        if f != self.animFrame:
            self._inertialize_into(f, lo, hi)
        else:
            self.lo, self.hi = lo, hi

    # --- attachment ----------------------------------------------------------
    def _attach_vase(self, qpos):
        """Lock the vase to the palm: keep the palm->vase pose of this frame."""
        palm_p, palm_q = self.armik.palm_pose(qpos)
        self.rel_pos = quat.inv_mul_vec(palm_q, self.vase_pos - palm_p)
        self.rel_quat = quat.mul(quat.inv(palm_q), self.vase_quat)
        self.held = True

    def _carry_vase(self, qpos):
        palm_p, palm_q = self.armik.palm_pose(qpos)
        self.vase_pos = palm_p + quat.mul_vec(palm_q, self.rel_pos)
        self.vase_quat = quat.mul(palm_q, self.rel_quat)

    # --- one real-time frame --------------------------------------------------
    def step(self, desiredVel, desiredFace):
        """Advance one frame. Returns the world qpos (36,) to play back; the
        vase pose is kept on self.vase_pos / self.vase_quat / self.held."""
        desiredVel = np.asarray(desiredVel, float)
        self._predict_trajectory(desiredVel, desiredFace)
        self._update_pick_err()
        self._maybe_trigger_pick()

        if self.searchTimer <= 0.0 and self.pick_locked == 0:
            self._search_loco()
            self.searchTimer = C.SEARCH_TIME

        self.animFrame = int(np.clip(self.animFrame + 1, self.lo, self.hi - 1))
        self.searchTimer -= DT
        riding = self.pick_locked > 0
        if riding:
            self.pick_locked -= 1
        elif self.animFrame >= self.hi - 2:
            self.searchTimer = 0.0
        f = self.animFrame

        # Root update: always relative -- integrate the matched clip's smooth
        # root velocity rotated into the live heading.
        _, _, self.rootAcc = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel,
            C.ROT_HALFLIFE, DT)
        qh_clip = yaw_quat(self.simThetaDB[f])
        clipVelLocal = quat.inv_mul_vec(qh_clip, self.simVelDB[f])
        self.rootVel = quat.mul_vec(self.rootRot, clipVelLocal)
        self.rootAng = np.array([0.0, 0.0, self.yawRateDB[f]])
        self.rootPos = self.rootPos + self.rootVel * DT
        self.rootYaw = self.rootYaw + self.yawRateDB[f] * DT
        self.rootRot = yaw_quat(self.rootYaw)

        # The grab gate: once, at the idle->reach boundary.
        if (riding and not self.grab_checked
                and self.phase[f] >= C.PHASE_REACH):
            self.grab_checked = True
            if C.POST_PROCESSING and not self._grab_ok():
                self._end_skill()
                riding = False
                f = self.animFrame

        if riding and self.pick_locked == 0:
            self._end_skill()
            f = self.animFrame

        # Inertialize joints + pelvis-local offset, reconstruct the pose.
        self.offDof, self.offDofVel = DecaySpringDamperPosition(
            self.offDof, self.offDofVel, C.INERT_HALFLIFE, DT)
        self.offPP, self.offPPVel = DecaySpringDamperPosition(
            self.offPP, self.offPPVel, C.INERT_HALFLIFE, DT)
        self.offPR, self.offPAng = DecaySpringDamperRotation(
            self.offPR, self.offPAng, C.INERT_HALFLIFE, DT)

        dofOut = self.dof[f] + self.offDof
        pelvLocalPos = self.plpDB[f] + self.offPP
        pelvLocalRot = quat.mul(self.offPR, self.prDB[f])
        pelvWorldPos = self.rootPos + quat.mul_vec(self.rootRot, pelvLocalPos)
        pelvWorldRot = quat.mul(self.rootRot, pelvLocalRot)

        qpos = np.empty(36)
        qpos[0:3] = pelvWorldPos
        qpos[3:7] = pelvWorldRot
        qpos[7:] = dofOut

        # Reach-phase arm IK overlay: ramps in over the reach, holds through
        # the grasp, then fades while the vase already rides the palm.
        in_reach = (C.POST_PROCESSING and self.state == C.SKILL_PICK
                    and C.PHASE_REACH <= self.phase[f] <= C.PHASE_GRASP)
        target_w = 1.0 if in_reach else 0.0
        self.ikW += (target_w - self.ikW) * (1.0 - 0.5 ** (DT / C.IK_BLEND_HALFLIFE))
        if in_reach:
            self.ikDelta = self.armik.solve(qpos, self.handPosW[f],
                                            self.handDirW[f], self.ikDelta)
        elif self.ikW <= 1e-3:
            self.ikW = 0.0
            self.ikDelta = np.zeros(7)
        if self.ikW > 0.0:
            qpos[C.ARM_QPOS] = qpos[C.ARM_QPOS] + self.ikW * self.ikDelta

        # The vase: lock to the palm when the contact flag turns on, then
        # follow the hand.
        if (self.state == C.SKILL_PICK and not self.held
                and self.contact[f] > 0.5):
            self._attach_vase(qpos)
        if self.held:
            self._carry_vase(qpos)
        return qpos

"""Real-time motion matching for walking, plus the pick skill on B.

The locomotion core is the same as motionmatching-g1-door: a smoothed
"simulation root" the matcher tracks + integrates, per-clip KD-tree search,
inertialized cuts. The pick runs as a small state machine:

    LOCOMOTION --B--> MOVE-TO-PICK --arrived--> PICK --clip end--> LOCOMOTION

MOVE-TO-PICK is still motion matching, but the command is made here: walk
toward the clip's recorded stance (its root pose at the pick entry frame),
so the trajectory query is built from the current state and the pick spot.
Once the robot is close enough, PICK plays the clip to the end with no
re-matching; the last root offset to the exact stance blends away during
the idle frames, so the hand lands on the vase. When the clip's contact
flag turns on, the vase snaps onto the right palm with the recorded grip
pose and follows the hand from then on.
"""
import numpy as np
from scipy.spatial import cKDTree

import config as C
import quat
from arm_fk import ArmFK
from features import build_db, yaw_quat, FORWARD, HORIZONS, FPS
from springs import (DecaySpringDamperPosition, DecaySpringDamperRotation,
                     TrajectorySpringPosition, TrajectorySpringRotation)

DT = C.DT
NDOF = 29
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
STATE_MOVE = 2                       # controller-only; clips are LOCO or PICK


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


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
        self.clip_id = lib["clip_id"]
        self.skill = lib["skill"]
        self.phase = lib["phase"]
        self.contact = lib["contact"]
        self.vase_rest = lib["vase_pos"].astype(np.float64)   # world, vase base
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

        # The pick clip: entered at its first frame, played to the end. The
        # idle second at the start is the window where the root offset from
        # walking blends away.
        pick_rows = np.where(self.skill == C.SKILL_PICK)[0]
        lo, hi = self._clip_bounds(int(pick_rows[0]))
        self.pick_entry = lo
        self.pick_lo, self.pick_hi = lo, hi

        # The recorded stance: where move-to-pick walks to.
        self.stance_xy = self.simPosDB[self.pick_entry][:2].copy()
        self.stance_yaw = float(self.simThetaDB[self.pick_entry])

        # The grip: palm -> vase pose at the clip's own grab frame. When the
        # contact flag turns on, the vase snaps to this pose on the live palm.
        self.armfk = ArmFK(lib)
        grab = lo + int(np.argmax(self.contact[lo:hi] > 0.5))
        palm_p, palm_q = self.armfk.palm_pose(lib["qpos"][grab].astype(np.float64))
        self.snap_pos = quat.inv_mul_vec(palm_q, self.vase_rest - palm_p)
        self.snap_quat = quat.inv(palm_q)     # the vase stands upright at rest
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
        self.pick_pending = False
        self.pick_locked = 0
        self.move_timer = 0.0
        self.on_rail = False
        self.move_target = self.stance_xy.copy()
        # The command actually fed to the matcher (shown by the viewer).
        self.cmdVel = np.zeros(3)
        self.cmdFace = np.zeros(3)
        self.pelvis_speed = 0.0
        self._prev_pelvis = None
        # The vase: on the shelf until the grab, then stuck to the palm.
        self.held = False
        self.vase_pos = self.vase_rest.copy()
        self.vase_quat = IDENTITY.copy()
        self.Tpos = np.tile(self.rootPos, (len(HORIZONS), 1))
        self.Tdir = np.tile(self.desiredDir, (len(HORIZONS), 1))

    def _clip_bounds(self, frame):
        r = int(np.searchsorted(self.starts, frame, "right") - 1)
        return int(self.starts[r]), int(self.stops[r])

    @property
    def cur(self):
        return self.animFrame

    def state_name(self):
        return {C.SKILL_LOCO: "LOCOMOTION", C.SKILL_PICK: "PICK",
                STATE_MOVE: "MOVE-TO-PICK"}[self.state]

    # --- trigger ------------------------------------------------------------
    def trigger_pick(self):
        """B: walk to the pick stance and play the clip. Pressing B again
        while walking there cancels."""
        if self.state == STATE_MOVE:
            self.state = C.SKILL_LOCO
            return
        if self.pick_locked > 0 or self.state != C.SKILL_LOCO or self.held:
            return
        self.pick_pending = True

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

    # --- move-to-pick --------------------------------------------------------
    def _steer_to_stance(self):
        """Walk onto the rail behind the stance first, then straight in
        along the stance heading -- plain forward walking, which the data
        has, instead of sideways shuffling, which it does not."""
        rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        rel = self.rootPos[0:2] - self.stance_xy
        along = float(rel @ rail)
        n = float(np.linalg.norm(rel - along * rail))
        # Latch onto the rail leg; only fall back off it on a big miss.
        if not self.on_rail:
            if along < -0.25 and n < 0.25:
                self.on_rail = True
                self.move_timer = 0.0      # fresh time budget for the rail leg
        elif n > 0.45 or along > 0.1:
            self.on_rail = False
        target = self.stance_xy if self.on_rail else self.stance_xy - 0.6 * rail
        self.move_target = target
        to = target - self.rootPos[0:2]
        dist = float(np.linalg.norm(to))
        vel = np.zeros(3)
        if dist > 1e-6:
            speed = float(np.clip(1.8 * dist, 0.25, 1.2))
            vel[0:2] = to / dist * speed
        if self.on_rail or dist < 1e-6:
            face = np.array([rail[0], rail[1], 0.0])
        else:
            face = vel / (np.linalg.norm(vel) + 1e-9)
        return vel, face

    def _at_stance(self):
        """Arrived: very close, or walking the rail came to a stop nearby."""
        dist = float(np.linalg.norm(self.stance_xy - self.rootPos[:2]))
        dyaw = abs(wrap_angle(self.stance_yaw - self.rootYaw))
        if dyaw >= C.MOVE_ARRIVE_YAW:
            return False
        if dist < C.MOVE_ARRIVE_NEAR:
            return True
        return (self.on_rail and self.move_timer > 1.0
                and dist < C.MOVE_ARRIVE_DIST and self.pelvis_speed < 0.15)

    def _enter_pick(self):
        """Cut into the pick clip. The stance offset left at this point is
        small; the vase snap absorbs it at the grab."""
        self._inertialize_into(self.pick_entry, self.pick_lo, self.pick_hi)
        self.state = C.SKILL_PICK
        self.pick_locked = self.pick_hi - 1 - self.pick_entry
        self.searchTimer = C.SEARCH_TIME

    def _end_skill(self):
        """Pick finished: search locomotion and blend back."""
        self.pick_locked = 0
        f, lo, hi = self._search_trees(self.loco_trees, self._query(),
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

    def _blend_goal_taps(self, desiredVel, desiredFace):
        """Bend the future taps onto the straight line to the current move
        target, so the query (and the red path) follows the command line.
        On the rail leg the taps also stop at the stance."""
        to = self.move_target - self.rootPos[0:2]
        dist = float(np.linalg.norm(to))
        if dist >= C.MOVE_GOAL_DIST or dist < 1e-6:
            return
        w = 1.0 - dist / C.MOVE_GOAL_DIST
        line_dir = to / dist
        speed = float(np.linalg.norm(desiredVel))
        for k, t in enumerate(self.Ttimes):
            s = speed * float(t)
            if self.on_rail:
                s = min(s, dist)          # arrive at the stance and stop
            line = self.Tpos[k].copy()
            line[0:2] = self.rootPos[0:2] + line_dir * s
            self.Tpos[k] = (1.0 - w) * self.Tpos[k] + w * line
            d = (1.0 - w) * self.Tdir[k] + w * desiredFace
            self.Tdir[k] = d / (np.linalg.norm(d) + 1e-9)

    def _query(self):
        d = self.db["dbs"]["loco"]
        f = self.animFrame
        qh = yaw_quat(self.rootYaw)
        parts = [self.rawXpos[f], self.rawXvel[f],
                 quat.inv_mul_vec(qh, self.Tpos - self.rootPos)[:, 0:2].ravel(),
                 quat.inv_mul_vec(qh, self.Tdir)[:, 0:2].ravel()]
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
        f, lo, hi = self._search_trees(self.loco_trees, self._query(),
                                       self.Xloco, True)
        if f != self.animFrame:
            self._inertialize_into(f, lo, hi)
        else:
            self.lo, self.hi = lo, hi

    # --- one real-time frame --------------------------------------------------
    def step(self, desiredVel, desiredFace):
        """Advance one frame. Returns the world qpos (36,) to play back; the
        vase pose is kept on self.vase_pos / self.vase_quat / self.held."""
        if self.pick_pending:
            self.pick_pending = False
            self.state = STATE_MOVE
            self.move_timer = 0.0

        # Move-to-pick drives itself: the player command is replaced by the
        # walk toward the recorded stance.
        if self.state == STATE_MOVE:
            self.move_timer += DT
            desiredVel, desiredFace = self._steer_to_stance()
            if self._at_stance():
                self._enter_pick()
                desiredVel = np.zeros(3)
                desiredFace = np.zeros(3)
            elif self.move_timer > C.MOVE_TIMEOUT:
                self.state = C.SKILL_LOCO

        desiredVel = np.asarray(desiredVel, float)
        self.cmdVel = desiredVel.copy()
        self.cmdFace = np.asarray(desiredFace, float).copy()
        self._predict_trajectory(desiredVel, desiredFace)
        if self.state == STATE_MOVE:
            self._blend_goal_taps(desiredVel, np.asarray(desiredFace, float))

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

        # Measured body speed, used to gate the offset blend next frame.
        if self._prev_pelvis is not None:
            self.pelvis_speed = float(
                np.linalg.norm(qpos[0:2] - self._prev_pelvis)) / DT
        self._prev_pelvis = qpos[0:2].copy()

        # The vase: snap onto the palm when the contact flag turns on, then
        # follow the hand with the recorded grip pose.
        if (self.state == C.SKILL_PICK and not self.held
                and self.contact[f] > 0.5):
            self.held = True
        if self.held:
            palm_p, palm_q = self.armfk.palm_pose(qpos)
            self.vase_pos = palm_p + quat.mul_vec(palm_q, self.snap_pos)
            self.vase_quat = quat.mul(palm_q, self.snap_quat)
        return qpos

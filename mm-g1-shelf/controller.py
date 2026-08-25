"""Real-time motion matching for walking, plus the pick skill on B.

The locomotion core is the same as motionmatching-g1-door: a smoothed
"simulation root" the matcher tracks + integrates, per-clip KD-tree search,
inertialized cuts. The pick runs as a small state machine:

    LOCOMOTION --B--> MOVE-TO-PICK --arrived--> PICK --clip end--> LOCOMOTION

MOVE-TO-PICK is still motion matching, but the command is made here: walk
a planned route to the clip's recorded stance (straight to a way-in point
behind it, around a rounded corner, then in along the stance heading), with
the future taps read straight off that route. On the final leg the root is
pinned to the rail, and the route aims past the stance so the walk never
slows into the dead zone; PICK starts at the stance crossing and plays the
clip to the end with no re-matching. The bottle welds onto the right palm
as soon as the live grip pose touches it (WELD_RADIUS, with the clip's
contact frame as fallback) and follows the hand from then on.
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

        # The pick clip: entered at its first frame, played to the end.
        pick_rows = np.where(self.skill == C.SKILL_PICK)[0]
        lo, hi = self._clip_bounds(int(pick_rows[0]))
        self.pick_entry = lo
        self.pick_lo, self.pick_hi = lo, hi

        # The recorded stance: where move-to-pick walks to. The way-in
        # point sits 0.6 m behind it on the rail.
        self.stance_xy = self.simPosDB[self.pick_entry][:2].copy()
        self.stance_yaw = float(self.simThetaDB[self.pick_entry])
        self.route_wp = self.stance_xy - 0.6 * np.array(
            [np.cos(self.stance_yaw), np.sin(self.stance_yaw)])

        # The grip: palm -> vase pose at the clip's own grab frame, pushed
        # WELD_STANDOFF out along the recorded approach so the palm rests on
        # the bottle's boundary instead of its center line.
        self.armfk = ArmFK(lib)
        grab = lo + int(np.argmax(self.contact[lo:hi] > 0.5))
        palm_p, palm_q = self.armfk.palm_pose(lib["qpos"][grab].astype(np.float64))
        prev_p, _ = self.armfk.palm_pose(lib["qpos"][grab - 8].astype(np.float64))
        appr = palm_p - prev_p
        appr[2] = 0.0                         # the bottle stands upright
        appr /= np.linalg.norm(appr)
        grip_rest = self.vase_rest + C.WELD_STANDOFF * appr
        self.snap_pos = quat.inv_mul_vec(palm_q, grip_rest - palm_p)
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
        self.route_pts = [self.rootPos[0:2].copy(), self.stance_xy.copy()]
        # The command actually fed to the matcher (shown by the viewer).
        self.cmdVel = np.zeros(3)
        self.cmdFace = np.zeros(3)
        # The vase: on the shelf until the grab, then stuck to the palm.
        self.held = False
        self.vase_pos = self.vase_rest.copy()
        self.vase_quat = IDENTITY.copy()
        # Weld offset: rest pose minus grip pose at the weld moment,
        # inertialized to zero while held.
        self.offVase = np.zeros(3); self.offVaseVel = np.zeros(3)
        self.offVaseRot = IDENTITY.copy(); self.offVaseAng = np.zeros(3)
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
    def _route_points(self):
        """The planned route as a polyline from the robot to the overshoot
        target: straight to the way-in point, a rounded corner there, then
        straight in along the rail. The corner arc keeps the heading
        turning continuously -- no sharp bend, no sudden yaw."""
        rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        end = self.stance_xy + C.MOVE_OVERSHOOT * rail
        p0 = self.rootPos[0:2]
        if self.on_rail:
            return [p0, end]
        wp = self.route_wp
        d1 = wp - p0
        L1 = float(np.linalg.norm(d1))
        if L1 < 1e-6:
            return [p0, end]
        d1 = d1 / L1
        ang = float(np.arccos(np.clip(d1 @ rail, -1.0, 1.0)))
        if ang < 0.15:
            return [p0, end]
        # Round the corner at the way-in point with radius ~0.25 m; cap the
        # fillet so sharp approach angles keep a real straight leg.
        t = min(0.25 * np.tan(ang / 2.0), 0.3, 0.6 * L1,
                0.5 * float(np.linalg.norm(end - wp)))
        A = wp - d1 * t
        B = wp + rail * t
        corner = [(1 - s) ** 2 * A + 2 * (1 - s) * s * wp + s * s * B
                  for s in np.linspace(0.0, 1.0, 9)[1:-1]]
        return [p0, A] + corner + [B, end]

    def _steer_to_stance(self):
        """Walk the planned route: toward a look-ahead point on the curve,
        facing the travel direction -- plain forward walking, which the
        data has, instead of sideways shuffling, which it does not."""
        rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        rel = self.rootPos[0:2] - self.stance_xy
        along = float(rel @ rail)
        n = float(np.linalg.norm(rel - along * rail))
        # Latch onto the final leg once the curve has merged with the rail;
        # only fall back off it on a big miss.
        if not self.on_rail:
            if along < -0.1 and n < 0.15:
                self.on_rail = True
                self.move_timer = 0.0      # fresh time budget for the last leg
        elif n > 0.45 or along > 0.1:
            self.on_rail = False

        route = self._route_points()
        self.route_pts = route
        # Route length and the look-ahead point ~0.45 m down the curve.
        look = route[-1]
        total = 0.0
        acc = 0.0
        prev = route[0]
        found = False
        for p in route[1:]:
            seg = float(np.linalg.norm(p - prev))
            total += seg
            if not found:
                acc += seg
                if acc >= 0.45:
                    look = p
                    found = True
            prev = p

        to = look - self.rootPos[0:2]
        dist = float(np.linalg.norm(to))
        vel = np.zeros(3)
        if dist > 1e-6:
            speed = float(np.clip(1.8 * total, 0.25, 1.2))
            vel[0:2] = to / dist * speed
        face = (vel / (np.linalg.norm(vel) + 1e-9) if dist > 1e-6
                else np.array([rail[0], rail[1], 0.0]))
        return vel, face

    def _at_stance(self):
        """Arrived: already standing at the stance, or walking the rail and
        just crossed the stance plane."""
        rel = self.rootPos[0:2] - self.stance_xy
        dist = float(np.linalg.norm(rel))
        dyaw = abs(wrap_angle(self.stance_yaw - self.rootYaw))
        if not self.on_rail:
            # B pressed while already standing at the stance.
            return dist < C.MOVE_ARRIVE_NEAR and dyaw < C.MOVE_ARRIVE_YAW
        rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        return float(rel @ rail) > -0.02

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

    def _path_taps(self):
        """The future taps read straight off the planned route: walk the
        remaining path at the approach speed profile and sample the
        horizons. We know the trajectory we want, so the query asks for
        exactly it."""
        pts = [p.copy() for p in self.route_pts[1:]]
        pos = self.rootPos[0:2].copy()
        heading = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
        k = 0
        for i in range(1, int(HORIZONS[-1]) + 1):
            rem, prev = 0.0, pos
            for p in pts:
                rem += float(np.linalg.norm(p - prev))
                prev = p
            adv = float(np.clip(1.8 * rem, 0.0, 1.2)) * DT
            while adv > 1e-9 and pts:
                seg = pts[0] - pos
                L = float(np.linalg.norm(seg))
                if L < 1e-9:
                    pts.pop(0)
                    continue
                heading = seg / L
                if adv < L:
                    pos = pos + heading * adv
                    adv = 0.0
                else:
                    pos = pts.pop(0)
                    adv -= L
            if k < len(HORIZONS) and i == int(HORIZONS[k]):
                self.Tpos[k] = np.array([pos[0], pos[1], 0.0])
                self.Tdir[k] = np.array([heading[0], heading[1], 0.0])
                k += 1

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
    def _search_trees(self, trees, Xq, Xself, with_bias):
        bestF, bestLo, bestHi = self.animFrame, self.lo, self.hi
        if with_bias and self.animFrame < self.hi - HORIZONS[-1]:
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
            self._path_taps()

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

        # Path snap: on the final leg the root is pinned to the rail, in
        # position and in heading -- the cross-track and yaw parts of the
        # matched motion are projected out.
        if self.state == STATE_MOVE and self.on_rail:
            to = self.stance_xy - self.rootPos[0:2]
            if float(np.linalg.norm(to)) < C.SNAP_RADIUS:
                rail = np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw)])
                cross = to - float(to @ rail) * rail
                a = 1.0 - 0.5 ** (DT / C.SNAP_HALFLIFE)
                self.rootPos[0:2] += a * cross
                self.rootYaw += a * wrap_angle(self.stance_yaw - self.rootYaw)
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

        # The vase: weld onto the palm when the live grip pose touches the
        # resting bottle (recorded contact frame as fallback). The offset
        # left at that moment is inertialized away, so the bottle is carried
        # off from where it stood instead of jumping to the hand.
        if self.state == C.SKILL_PICK and not self.held:
            palm_p, palm_q = self.armfk.palm_pose(qpos)
            grip = palm_p + quat.mul_vec(palm_q, self.snap_pos)
            if (np.linalg.norm(grip - self.vase_pos) < C.WELD_RADIUS
                    or self.contact[f] > 0.5):
                self.held = True
                self.offVase = self.vase_pos - grip
                self.offVaseRot = quat.mul(
                    self.vase_quat, quat.inv(quat.mul(palm_q, self.snap_quat)))
        if self.held:
            palm_p, palm_q = self.armfk.palm_pose(qpos)
            self.offVase, self.offVaseVel = DecaySpringDamperPosition(
                self.offVase, self.offVaseVel, C.WELD_HALFLIFE, DT)
            self.offVaseRot, self.offVaseAng = DecaySpringDamperRotation(
                self.offVaseRot, self.offVaseAng, C.WELD_HALFLIFE, DT)
            self.vase_pos = (palm_p + quat.mul_vec(palm_q, self.snap_pos)
                             + self.offVase)
            self.vase_quat = quat.mul(self.offVaseRot,
                                      quat.mul(palm_q, self.snap_quat))
        return qpos

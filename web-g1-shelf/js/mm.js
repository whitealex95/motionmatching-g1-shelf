// Real-time motion matching in the browser -- a port of
// mm-g1-shelf/controller.py (GenoView smoothed-sim-root locomotion + the
// B pick skill). Reads the databases exported by export_web_data.py.
//
//   LOCOMOTION --B--> MOVE-TO-PICK --arrived--> PICK --clip end--> LOCOMOTION
//
// MOVE-TO-PICK walks a planned route to the recorded stance (way-in point,
// rounded corner, then in along the stance heading, aiming a bit past it);
// on the final leg the root is pinned to the rail. PICK plays the clip with
// no re-matching; the bottle welds onto the right palm as soon as the live
// grip pose touches it (contact frame as fallback), the leftover offset is
// inertialized away, and it follows the hand from then on.

import { quat, v3 } from './quat.js';
import { fk } from './fk.js';

const FORWARD = [1, 0, 0];
const IDENTITY = [1, 0, 0, 0];
const STATE_MOVE = 2;                 // controller-only; clips are LOCO or PICK

// ---- DB loader: typed-array views into mm.bin per the mm.json header ----
export function loadDB(meta, buf) {
  const A = {};
  for (const [name, h] of Object.entries(meta.arrays)) {
    const TA = h.dtype === 'int32' ? Int32Array : Float32Array;
    const count = h.shape.reduce((a, b) => a * b, 1);
    A[name] = count ? new TA(buf, h.offset, count) : new TA(0);
  }
  return A;
}

// ---- spring + inertialization helpers (port of springs.py) ----
const damp = (hl) => (4.0 * 0.69314718056) / (hl + 1e-5);

function decayPos(x, v, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt), xo = [], vo = [];
  for (let i = 0; i < x.length; i++) {
    const j1 = v[i] + x[i] * y;
    xo[i] = e * (x[i] + j1 * dt);
    vo[i] = e * (v[i] - j1 * y * dt);
  }
  return [xo, vo];
}
function decayRot(x, v, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt);
  const j0 = quat.toScaledAngleAxis(x);
  const j1 = [v[0] + j0[0] * y, v[1] + j0[1] * y, v[2] + j0[2] * y];
  const q = quat.fromScaledAngleAxis([e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]);
  return [q, [e * (v[0] - j1[0] * y * dt), e * (v[1] - j1[1] * y * dt), e * (v[2] - j1[2] * y * dt)]];
}
function trajPos(pos, vel, acc, dvel, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt), P = [], V = [], Ac = [];
  for (let i = 0; i < 3; i++) {
    const j0 = vel[i] - dvel[i], j1 = acc[i] + j0 * y;
    P[i] = e * ((-j1) / (y * y) + (-j0 - j1 * dt) / y) + j1 / (y * y) + j0 / y + dvel[i] * dt + pos[i];
    V[i] = e * (j0 + j1 * dt) + dvel[i];
    Ac[i] = e * (acc[i] - j1 * y * dt);
  }
  return [P, V, Ac];
}
function trajRot(rot, ang, dRot, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt);
  const j0 = quat.toScaledAngleAxis(quat.abs(quat.mul_inv(rot, dRot)));
  const j1 = [ang[0] + j0[0] * y, ang[1] + j0[1] * y, ang[2] + j0[2] * y];
  const q = quat.mul(quat.fromScaledAngleAxis([e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]), dRot);
  return [q, [e * (ang[0] - j1[0] * y * dt), e * (ang[1] - j1[1] * y * dt), e * (ang[2] - j1[2] * y * dt)]];
}

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrapAngle = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

export class MotionMatcher {
  // meta + A from mm.json/mm.bin; bodies = model.json body list (used for
  // the palm FK that keeps the vase glued to the hand).
  constructor(meta, A, bodies) {
    this.A = A;
    this.bodies = bodies;
    this.fps = meta.fps; this.DT = 1 / meta.fps;
    this.H = Math.max(...meta.horizons);
    this.Ttimes = meta.horizons.map((h) => h / meta.fps);
    this.horizons = meta.horizons;
    this.MAX_SPEED = meta.max_speed; this.WALK_SCALE = meta.walk_scale;
    this.SEARCH_TIME = meta.search_time; this.CURRENT_BIAS = meta.current_bias;
    this.INERT = meta.inert_halflife; this.VEL_HL = meta.vel_halflife; this.ROT_HL = meta.rot_halflife;
    this.LOCO = meta.skill_loco; this.PICK = meta.skill_pick;
    this.OVERSHOOT = meta.move_overshoot;
    this.ARRIVE_NEAR = meta.move_arrive_near;
    this.ARRIVE_YAW = meta.move_arrive_yaw;
    this.MOVE_TIMEOUT = meta.move_timeout;
    this.SNAP_RADIUS = meta.snap_radius; this.SNAP_HL = meta.snap_halflife;
    this.WELD_RADIUS = meta.weld_radius; this.WELD_HL = meta.weld_halflife;
    this.PALM = meta.palm_offset;
    this.ARM0 = meta.arm_qpos_start;
    this.pickEntry = meta.pick_entry;
    this.pickLo = meta.pick_lo; this.pickHi = meta.pick_hi;
    this.stanceXY = meta.stance_xy; this.stanceYaw = meta.stance_yaw;
    this.routeWpFixed = meta.route_wp;
    this.vaseRest = meta.vase_rest;
    this.snapPos = meta.snap_pos; this.snapQuat = meta.snap_quat;
    this.clipNames = meta.clip_names;
    this.wrist = bodies.findIndex((b) => b.name === 'right_wrist_yaw_link');

    this.starts = A.starts; this.stops = A.stops;
    this.Xloco = A.Xloco;
    this.locoOffset = A.locoOffset; this.locoScale = A.locoScale;
    this.locoSegs = Array.from(A.search_clips, (ci) => [A.starts[ci], A.stops[ci]]);
    this.reset();
  }

  // row accessors (return plain arrays)
  _row(arr, d, f) { const b = f * d, r = new Array(d); for (let i = 0; i < d; i++) r[i] = arr[b + i]; return r; }
  dof(f) { return this._row(this.A.dof, 29, f); }
  dofVel(f) { return this._row(this.A.dofVel, 29, f); }
  simPos(f) { return this._row(this.A.simPos, 3, f); }
  simVel(f) { return this._row(this.A.simVel, 3, f); }
  simTheta(f) { return this.A.simTheta[f]; }
  yawRate(f) { return this.A.yawRate[f]; }
  pelvLocalPos(f) { return this._row(this.A.pelvLocalPos, 3, f); }
  pelvLocalVel(f) { return this._row(this.A.pelvLocalVel, 3, f); }
  pelvLocalRot(f) { return this._row(this.A.pelvLocalRot, 4, f); }
  pelvLocalAng(f) { return this._row(this.A.pelvLocalAng, 3, f); }
  rawXpos(f) { return this._row(this.A.rawXpos, 6, f); }
  rawXvel(f) { return this._row(this.A.rawXvel, 9, f); }
  contact(f) { return this.A.contact[f]; }

  _clipOf(f) { let r = 0; for (let i = 0; i < this.starts.length; i++) { if (this.starts[i] <= f) r = i; else break; } return r; }
  _clipBounds(f) { const r = this._clipOf(f); return [this.starts[r], this.stops[r]]; }

  reset() {
    const sf = Math.min(this.stops[0] - 1, this.starts[0] + 30);
    [this.lo, this.hi] = this._clipBounds(sf);
    this.animFrame = sf;
    this.state = this.LOCO;
    // Start at the world origin facing +x; the shelf sits ahead at the
    // shifted spot the clip recorded.
    this.rootPos = [0, 0, 0];
    this.rootVel = [0, 0, 0]; this.rootAcc = [0, 0, 0]; this.rootAng = [0, 0, 0];
    this.rootYaw = 0;
    this.rootRot = quat.yaw(this.rootYaw);
    this.desiredDir = quat.mulVec(this.rootRot, FORWARD);
    this.offDof = new Array(29).fill(0); this.offDofVel = new Array(29).fill(0);
    this.offPP = [0, 0, 0]; this.offPPVel = [0, 0, 0];
    this.offPR = IDENTITY.slice(); this.offPAng = [0, 0, 0];
    this.searchTimer = 0;
    this.pickPending = false; this.pickLocked = 0;
    this.moveTimer = 0;
    this.onRail = false;
    this.routePts = [[0, 0], this.stanceXY.slice()];
    this.cmdVel = [0, 0, 0]; this.cmdFace = [0, 0, 0];
    // The vase: on the shelf until the grab, then stuck to the palm.
    this.held = false;
    // Weld offset: rest pose minus grip pose at the weld moment,
    // inertialized to zero while held.
    this.offVase = [0, 0, 0]; this.offVaseVel = [0, 0, 0];
    this.offVaseRot = [1, 0, 0, 0]; this.offVaseAng = [0, 0, 0];
    this.vasePos = this.vaseRest.slice();
    this.vaseQuat = IDENTITY.slice();
    this.Tpos = [this.rootPos, this.rootPos, this.rootPos];
    this.Tdir = [this.desiredDir, this.desiredDir, this.desiredDir];
  }

  // ---- public state / triggers ----
  // B: walk to the pick stance and play the clip. B again while walking
  // there cancels.
  triggerPick() {
    if (this.state === STATE_MOVE) { this.state = this.LOCO; return; }
    if (this.pickLocked > 0 || this.state !== this.LOCO || this.held) return;
    this.pickPending = true;
  }
  get cur() { return this.animFrame; }
  get riding() { return this.pickLocked > 0; }
  stateName() {
    if (this.state === STATE_MOVE) return 'MOVE-TO-PICK';
    return this.state === this.PICK ? 'PICK' : 'LOCOMOTION';
  }

  // ---- inertialized cut ----
  _inertInto(b, lo, hi) {
    const a = this.animFrame;
    const da = this.dof(a), db = this.dof(b), va = this.dofVel(a), vb = this.dofVel(b);
    for (let i = 0; i < 29; i++) { this.offDof[i] += da[i] - db[i]; this.offDofVel[i] += va[i] - vb[i]; }
    const pa = this.pelvLocalPos(a), pb = this.pelvLocalPos(b), qa = this.pelvLocalVel(a), qb = this.pelvLocalVel(b);
    for (let i = 0; i < 3; i++) { this.offPP[i] += pa[i] - pb[i]; this.offPPVel[i] += qa[i] - qb[i]; }
    this.offPR = quat.abs(quat.mul_inv(quat.mul(this.offPR, this.pelvLocalRot(a)), this.pelvLocalRot(b)));
    const aa = this.pelvLocalAng(a), ab = this.pelvLocalAng(b);
    for (let i = 0; i < 3; i++) this.offPAng[i] += aa[i] - ab[i];
    this.animFrame = b; this.lo = lo; this.hi = hi;
  }

  // ---- move-to-pick --------------------------------------------------------
  // The planned route as a polyline from the robot to the overshoot target:
  // straight to the way-in point, a rounded corner there, then straight in
  // along the rail.
  _routePoints() {
    const rail = [Math.cos(this.stanceYaw), Math.sin(this.stanceYaw)];
    const end = [this.stanceXY[0] + this.OVERSHOOT * rail[0],
                 this.stanceXY[1] + this.OVERSHOOT * rail[1]];
    const p0 = [this.rootPos[0], this.rootPos[1]];
    if (this.onRail) return [p0, end];
    const wp = this.routeWpFixed;
    let d1 = [wp[0] - p0[0], wp[1] - p0[1]];
    const L1 = Math.hypot(d1[0], d1[1]);
    if (L1 < 1e-6) return [p0, end];
    d1 = [d1[0] / L1, d1[1] / L1];
    const ang = Math.acos(clamp(d1[0] * rail[0] + d1[1] * rail[1], -1, 1));
    if (ang < 0.15) return [p0, end];
    const t = Math.min(0.25 * Math.tan(ang / 2), 0.3, 0.6 * L1,
                       0.5 * Math.hypot(end[0] - wp[0], end[1] - wp[1]));
    const A = [wp[0] - d1[0] * t, wp[1] - d1[1] * t];
    const B = [wp[0] + rail[0] * t, wp[1] + rail[1] * t];
    const pts = [p0, A];
    for (let k = 1; k < 8; k++) {
      const s = k / 8;
      pts.push([(1 - s) * (1 - s) * A[0] + 2 * (1 - s) * s * wp[0] + s * s * B[0],
                (1 - s) * (1 - s) * A[1] + 2 * (1 - s) * s * wp[1] + s * s * B[1]]);
    }
    pts.push(B, end);
    return pts;
  }

  // Walk the planned route: toward a look-ahead point on the curve, facing
  // the travel direction.
  _steerToStance() {
    const rail = [Math.cos(this.stanceYaw), Math.sin(this.stanceYaw)];
    const rel = [this.rootPos[0] - this.stanceXY[0], this.rootPos[1] - this.stanceXY[1]];
    const along = rel[0] * rail[0] + rel[1] * rail[1];
    const cx = rel[0] - along * rail[0], cy = rel[1] - along * rail[1];
    const n = Math.hypot(cx, cy);
    if (!this.onRail) {
      if (along < -0.1 && n < 0.15) {
        this.onRail = true;
        this.moveTimer = 0;             // fresh time budget for the last leg
      }
    } else if (n > 0.45 || along > 0.1) this.onRail = false;

    const route = this._routePoints();
    this.routePts = route;
    let look = route[route.length - 1];
    let total = 0, acc = 0, found = false;
    let prev = route[0];
    for (let i = 1; i < route.length; i++) {
      const p = route[i];
      const seg = Math.hypot(p[0] - prev[0], p[1] - prev[1]);
      total += seg;
      if (!found) {
        acc += seg;
        if (acc >= 0.45) { look = p; found = true; }
      }
      prev = p;
    }

    const to = [look[0] - this.rootPos[0], look[1] - this.rootPos[1]];
    const dist = Math.hypot(to[0], to[1]);
    let vel = [0, 0, 0];
    if (dist > 1e-6) {
      const speed = clamp(1.8 * total, 0.25, 1.2);
      vel = [to[0] / dist * speed, to[1] / dist * speed, 0];
    }
    const vn = Math.hypot(vel[0], vel[1]);
    const face = dist > 1e-6 ? [vel[0] / (vn + 1e-9), vel[1] / (vn + 1e-9), 0]
                             : [rail[0], rail[1], 0];
    return [vel, face];
  }

  // Arrived: already standing at the stance, or walking the rail and just
  // crossed the stance plane.
  _atStance() {
    const rel = [this.rootPos[0] - this.stanceXY[0], this.rootPos[1] - this.stanceXY[1]];
    const dist = Math.hypot(rel[0], rel[1]);
    const dyaw = Math.abs(wrapAngle(this.stanceYaw - this.rootYaw));
    if (!this.onRail) return dist < this.ARRIVE_NEAR && dyaw < this.ARRIVE_YAW;
    const rail = [Math.cos(this.stanceYaw), Math.sin(this.stanceYaw)];
    return rel[0] * rail[0] + rel[1] * rail[1] > -0.02;
  }

  _enterPick() {
    this._inertInto(this.pickEntry, this.pickLo, this.pickHi);
    this.state = this.PICK;
    this.pickLocked = this.pickHi - 1 - this.pickEntry;
    this.searchTimer = this.SEARCH_TIME;
  }

  _endSkill() {
    this.pickLocked = 0;
    const [f, lo, hi] = this._search(this.locoSegs, this._query(), this.Xloco, 27, false);
    this._inertInto(f, lo, hi);
    this.state = this.LOCO;
    this.searchTimer = this.SEARCH_TIME;
  }

  // ---- command springs ----
  _predictTrajectory(desiredVel, desiredFace) {
    if (v3.norm(desiredFace) > 0.01) this.desiredDir = v3.scale(desiredFace, 1 / v3.norm(desiredFace));
    else if (v3.norm(desiredVel) > 0.01) this.desiredDir = v3.scale(desiredVel, 1 / v3.norm(desiredVel));
    const desiredRot = quat.yaw(Math.atan2(this.desiredDir[1], this.desiredDir[0]));
    this.Tpos = []; this.Tdir = [];
    for (let k = 0; k < 3; k++) {
      const [P] = trajPos(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.VEL_HL, this.Ttimes[k]);
      this.Tpos.push(P);
      const [Q] = trajRot(this.rootRot, this.rootAng, desiredRot, this.ROT_HL, this.Ttimes[k]);
      this.Tdir.push(quat.mulVec(Q, FORWARD));
    }
  }

  // The future taps read straight off the planned route: walk the remaining
  // path at the approach speed profile and sample the horizons.
  _pathTaps() {
    const pts = this.routePts.slice(1).map((p) => p.slice());
    let pos = [this.rootPos[0], this.rootPos[1]];
    let heading = [Math.cos(this.stanceYaw), Math.sin(this.stanceYaw)];
    let k = 0;
    for (let i = 1; i <= this.H; i++) {
      let rem = 0;
      let prev = pos;
      for (const p of pts) { rem += Math.hypot(p[0] - prev[0], p[1] - prev[1]); prev = p; }
      let adv = clamp(1.8 * rem, 0, 1.2) * this.DT;
      while (adv > 1e-9 && pts.length) {
        const seg = [pts[0][0] - pos[0], pts[0][1] - pos[1]];
        const L = Math.hypot(seg[0], seg[1]);
        if (L < 1e-9) { pts.shift(); continue; }
        heading = [seg[0] / L, seg[1] / L];
        if (adv < L) {
          pos = [pos[0] + heading[0] * adv, pos[1] + heading[1] * adv];
          adv = 0;
        } else {
          pos = pts.shift();
          adv -= L;
        }
      }
      if (k < 3 && i === this.horizons[k]) {
        this.Tpos[k] = [pos[0], pos[1], 0];
        this.Tdir[k] = [heading[0], heading[1], 0];
        k += 1;
      }
    }
  }

  // ---- query assembly (build_db block order) ----
  _query() {
    const qh = quat.yaw(this.rootYaw);
    const q = this.rawXpos(this.animFrame).concat(this.rawXvel(this.animFrame));   // pose (15)
    for (let k = 0; k < 3; k++) {                                                  // traj pos (6)
      const dp = quat.invMulVec(qh, v3.sub(this.Tpos[k], this.rootPos));
      q.push(dp[0], dp[1]);
    }
    for (let k = 0; k < 3; k++) {                                                  // traj dir (6)
      const dd = quat.invMulVec(qh, this.Tdir[k]);
      q.push(dd[0], dd[1]);
    }
    for (let i = 0; i < q.length; i++) q[i] = (q[i] - this.locoOffset[i]) / this.locoScale[i];
    return q;
  }

  // ---- brute-force per-clip search (KD-tree equivalent) ----
  _search(segs, Xq, Xmat, dim, withBias) {
    let bestF = this.animFrame, bestLo = this.lo, bestHi = this.hi;
    let best;
    if (withBias && this.animFrame < this.hi - this.H) {
      let s = 0; const b = this.animFrame * dim;
      for (let i = 0; i < dim; i++) { const d = Xq[i] - Xmat[b + i]; s += d * d; }
      best = Math.sqrt(s) - this.CURRENT_BIAS;
    } else best = Infinity;
    if (best <= 0) return [bestF, bestLo, bestHi];
    let bestSq = best * best;
    for (const [rs, re] of segs) {
      const lim = re - this.H;
      for (let f = rs; f < lim; f++) {
        const b = f * dim; let s = 0;
        for (let i = 0; i < dim; i++) { const d = Xq[i] - Xmat[b + i]; s += d * d; if (s >= bestSq) { s = -1; break; } }
        if (s >= 0) { bestSq = s; bestF = f; bestLo = rs; bestHi = re; }
      }
    }
    return [bestF, bestLo, bestHi];
  }

  _searchLoco() {
    const [f, lo, hi] = this._search(this.locoSegs, this._query(), this.Xloco, 27, true);
    if (f !== this.animFrame) this._inertInto(f, lo, hi); else { this.lo = lo; this.hi = hi; }
  }

  // World palm pose from the body-tree FK (right wrist + palm offset).
  _palmPose(qpos) {
    const { wp, wq } = fk(this.bodies, qpos);
    const p = v3.add(wp[this.wrist], quat.mulVec(wq[this.wrist], this.PALM));
    return [p, wq[this.wrist]];
  }

  // ---- one real-time frame ----
  // desiredVel [x,y,0] m/s; desiredFace [x,y,0] unit or zero. Returns qpos
  // (36,); the vase pose is kept on this.vasePos / this.vaseQuat / this.held.
  step(desiredVel, desiredFace) {
    if (this.pickPending) {
      this.pickPending = false;
      this.state = STATE_MOVE;
      this.moveTimer = 0;
    }

    // Move-to-pick drives itself: the player command is replaced by the
    // walk along the planned route.
    if (this.state === STATE_MOVE) {
      this.moveTimer += this.DT;
      [desiredVel, desiredFace] = this._steerToStance();
      if (this._atStance()) {
        this._enterPick();
        desiredVel = [0, 0, 0];
        desiredFace = [0, 0, 0];
      } else if (this.moveTimer > this.MOVE_TIMEOUT) {
        this.state = this.LOCO;
      }
    }

    this.cmdVel = desiredVel.slice(); this.cmdFace = desiredFace.slice();
    this._predictTrajectory(desiredVel, desiredFace);
    if (this.state === STATE_MOVE) this._pathTaps();

    if (this.pickLocked === 0 && this.searchTimer <= 0) {
      this._searchLoco();
      this.searchTimer = this.SEARCH_TIME;
    }

    this.animFrame = clamp(this.animFrame + 1, this.lo, this.hi - 1);
    this.searchTimer -= this.DT;
    let riding = this.pickLocked > 0;
    if (riding) this.pickLocked -= 1;
    else if (this.animFrame >= this.hi - 2) this.searchTimer = 0;
    let f = this.animFrame;

    // Root update: always relative -- integrate the matched clip's smooth
    // root velocity rotated into the live heading.
    const [, , acc] = trajPos(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.ROT_HL, this.DT);
    this.rootAcc = acc;
    const qhClip = quat.yaw(this.simTheta(f));
    const clipVelLocal = quat.invMulVec(qhClip, this.simVel(f));
    this.rootVel = quat.mulVec(this.rootRot, clipVelLocal);
    this.rootAng = [0, 0, this.yawRate(f)];
    this.rootPos = v3.add(this.rootPos, v3.scale(this.rootVel, this.DT));
    this.rootYaw += this.yawRate(f) * this.DT;

    // Path snap: on the final leg the root is pinned to the rail, in
    // position and in heading.
    if (this.state === STATE_MOVE && this.onRail) {
      const to = [this.stanceXY[0] - this.rootPos[0], this.stanceXY[1] - this.rootPos[1]];
      if (Math.hypot(to[0], to[1]) < this.SNAP_RADIUS) {
        const rail = [Math.cos(this.stanceYaw), Math.sin(this.stanceYaw)];
        const along = to[0] * rail[0] + to[1] * rail[1];
        const a = 1 - Math.pow(0.5, this.DT / this.SNAP_HL);
        this.rootPos[0] += a * (to[0] - along * rail[0]);
        this.rootPos[1] += a * (to[1] - along * rail[1]);
        this.rootYaw += a * wrapAngle(this.stanceYaw - this.rootYaw);
      }
    }
    this.rootRot = quat.yaw(this.rootYaw);

    if (riding && this.pickLocked === 0) {
      this._endSkill();
      f = this.animFrame;
    }

    [this.offDof, this.offDofVel] = decayPos(this.offDof, this.offDofVel, this.INERT, this.DT);
    [this.offPP, this.offPPVel] = decayPos(this.offPP, this.offPPVel, this.INERT, this.DT);
    [this.offPR, this.offPAng] = decayRot(this.offPR, this.offPAng, this.INERT, this.DT);

    const dof = this.dof(f), dofOut = new Array(29);
    for (let i = 0; i < 29; i++) dofOut[i] = dof[i] + this.offDof[i];
    const plp = v3.add(this.pelvLocalPos(f), this.offPP);
    const plr = quat.mul(this.offPR, this.pelvLocalRot(f));
    const pelvPos = v3.add(this.rootPos, quat.mulVec(this.rootRot, plp));
    const pelvRot = quat.mul(this.rootRot, plr);

    const qpos = new Float64Array(36);
    qpos[0] = pelvPos[0]; qpos[1] = pelvPos[1]; qpos[2] = pelvPos[2];
    qpos[3] = pelvRot[0]; qpos[4] = pelvRot[1]; qpos[5] = pelvRot[2]; qpos[6] = pelvRot[3];
    for (let i = 0; i < 29; i++) qpos[7 + i] = dofOut[i];

    // The vase: weld onto the palm when the live grip pose touches the
    // resting bottle (recorded contact frame as fallback). The offset left
    // at that moment is inertialized away, so the bottle is carried off
    // from where it stood instead of jumping to the hand.
    if (this.state === this.PICK && !this.held) {
      const [palmP, palmQ] = this._palmPose(qpos);
      const grip = v3.add(palmP, quat.mulVec(palmQ, this.snapPos));
      const d = Math.hypot(grip[0] - this.vasePos[0], grip[1] - this.vasePos[1],
                           grip[2] - this.vasePos[2]);
      if (d < this.WELD_RADIUS || this.contact(f) > 0.5) {
        this.held = true;
        this.offVase = v3.sub(this.vasePos, grip);
        this.offVaseRot = quat.mul(
          this.vaseQuat, quat.inv(quat.mul(palmQ, this.snapQuat)));
      }
    }
    if (this.held) {
      const [palmP, palmQ] = this._palmPose(qpos);
      [this.offVase, this.offVaseVel] = decayPos(this.offVase, this.offVaseVel, this.WELD_HL, this.DT);
      [this.offVaseRot, this.offVaseAng] = decayRot(this.offVaseRot, this.offVaseAng, this.WELD_HL, this.DT);
      this.vasePos = v3.add(v3.add(palmP, quat.mulVec(palmQ, this.snapPos)), this.offVase);
      this.vaseQuat = quat.mul(this.offVaseRot, quat.mul(palmQ, this.snapQuat));
    }
    return qpos;
  }
}

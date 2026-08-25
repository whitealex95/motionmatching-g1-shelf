// Interactive G1 shelf-pick demo (Three.js). Loads the exported database,
// runs the JS motion matcher (mm.js, a port of the Python controller) at a
// fixed 30 Hz, forward-kinematics the result, and draws the G1 with its full
// visual meshes. The IKEA BILLY shelf is static; the bottle is placed
// kinematically every frame (on the shelf, or stuck to the palm after the
// grab).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MotionMatcher, loadDB } from './mm.js';
import { fk } from './fk.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);   // MuJoCo is z-up

const DATA = './data';
const hud = document.getElementById('hud');
const setHud = (t) => { hud.textContent = t; };

async function loadJSON(u) { return (await fetch(u)).json(); }
async function loadBin(u) { return (await fetch(u)).arrayBuffer(); }

// Same normalization tools/convert_ikea_assets.py bakes into the MuJoCo
// OBJs: glTF Y-up -> Z-up, centered in X/Y, base at Z=0. The returned group
// can then take the same world pose as the MuJoCo body.
function toZUpBaseFrame(gltfScene) {
  gltfScene.rotation.x = Math.PI / 2;
  const holder = new THREE.Group();
  holder.add(gltfScene);
  holder.updateWorldMatrix(true, true);
  const bb = new THREE.Box3().setFromObject(gltfScene, true);
  gltfScene.position.set(-(bb.min.x + bb.max.x) / 2,
                         -(bb.min.y + bb.max.y) / 2, -bb.min.z);
  gltfScene.traverse((o) => {
    if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; }
  });
  return holder;
}

async function boot() {
  setHud('loading G1 model + motion database (~25 MB)...');
  const gltf = new GLTFLoader().setDRACOLoader(
    new DRACOLoader().setDecoderPath('./vendor/draco/'));
  const [model, meta, bin, meshMeta, meshBin, shelfJson, billyGltf, bottleGltf] =
    await Promise.all([
      loadJSON(`${DATA}/model.json`), loadJSON(`${DATA}/mm.json`), loadBin(`${DATA}/mm.bin`),
      loadJSON(`${DATA}/mesh.json`), loadBin(`${DATA}/mesh.bin`),
      loadJSON(`${DATA}/shelf.json`),
      gltf.loadAsync(`${DATA}/billy.glb`), gltf.loadAsync(`${DATA}/bottle.glb`),
    ]);
  const A = loadDB(meta, bin);
  const mm = new MotionMatcher(meta, A, model.bodies);
  window.mm = mm;                          // console access for debugging
  start(model.bodies, mm, meshMeta, meshBin, shelfJson, billyGltf, bottleGltf);
}

function start(bodies, mm, meshMeta, meshBuf, shelfJson, billyGltf, bottleGltf) {
  // ---- renderer / scene / camera (z-up) ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9a9286);
  scene.fog = new THREE.Fog(0x9a9286, 35, 110);

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 200);
  camera.up.set(0, 0, 1);
  camera.position.set(-1.6, -2.8, 1.9);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.8);
  controls.enablePan = false;

  // ---- lights + floor ----
  scene.add(new THREE.HemisphereLight(0xffffff, 0x554b40, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(4, -6, 8); sun.castShadow = true;
  sun.shadow.camera.top = 8; sun.shadow.camera.bottom = -8;
  sun.shadow.camera.left = -8; sun.shadow.camera.right = 8;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const cx = cv.getContext('2d');
  cx.fillStyle = '#7a7165'; cx.fillRect(0, 0, 256, 256);
  cx.fillStyle = '#5e564c'; cx.fillRect(0, 0, 128, 128); cx.fillRect(128, 128, 128, 128);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(100, 100);
  tex.anisotropy = 8;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
  floor.receiveShadow = true;
  scene.add(floor);

  // ---- G1 full mesh: one Group per body; FK moves the groups ----
  const robot = new THREE.Group();
  scene.add(robot);
  const bodyGroups = bodies.map(() => { const g = new THREE.Group(); robot.add(g); return g; });
  for (const gm of meshMeta.geoms) {
    const pos = new Float32Array(meshBuf, gm.vstart * 12, gm.vcount * 3);
    const idx = new Uint16Array(meshBuf, meshMeta.idx_byte_offset + gm.istart * 2, gm.icount);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(gm.rgba[0], gm.rgba[1], gm.rgba[2]),
      metalness: 0.55, roughness: 0.45, flatShading: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    bodyGroups[gm.body].add(mesh);
  }
  const _yAxis = new THREE.Vector3(0, 1, 0);
  const _Y2Z = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);

  // ---- the IKEA BILLY shelf (static) ----
  const billy = toZUpBaseFrame(billyGltf.scene);
  const bq = shelfJson.billy.quat;                       // wxyz
  billy.position.set(...shelfJson.billy.pos);
  billy.quaternion.set(bq[1], bq[2], bq[3], bq[0]);
  scene.add(billy);

  // ---- the bottle: one group, placed kinematically each frame ----
  const vase = toZUpBaseFrame(bottleGltf.scene);
  scene.add(vase);

  // ---- the pick-spot marker: a green disc + heading tick on the floor ----
  const marker = new THREE.Group();
  const green = new THREE.MeshBasicMaterial({
    color: 0x33cc4d, transparent: true, opacity: 0.5 });
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.008, 32), green);
  disc.position.set(mm.stanceXY[0], mm.stanceXY[1], 0.006);
  disc.quaternion.copy(_Y2Z);
  marker.add(disc);
  const tick = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.28, 6), green);
  const tdir = [Math.cos(mm.stanceYaw), Math.sin(mm.stanceYaw)];
  tick.position.set(mm.stanceXY[0] + 0.15 * tdir[0], mm.stanceXY[1] + 0.15 * tdir[1], 0.006);
  tick.quaternion.setFromUnitVectors(_yAxis, new THREE.Vector3(tdir[0], tdir[1], 0));
  marker.add(tick);
  scene.add(marker);

  // ---- gizmos: red trajectory taps + green planned route + blue command ----
  const gizmo = new THREE.Group(); scene.add(gizmo);
  const red = new THREE.MeshBasicMaterial({ color: 0xe21818 });
  const gizSph = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 10), red); gizmo.add(m); return m; });
  const gizStk = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), red); gizmo.add(m); return m; });
  const routeGeo = new THREE.BufferGeometry();
  const routeArr = new Float32Array(3 * 16);
  routeGeo.setAttribute('position', new THREE.BufferAttribute(routeArr, 3));
  const routeLine = new THREE.Line(routeGeo, new THREE.LineBasicMaterial({ color: 0x33cc4d }));
  gizmo.add(routeLine);
  const blue = new THREE.MeshBasicMaterial({ color: 0x3380f2 });
  const cmdArrow = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), blue);
  gizmo.add(cmdArrow);

  // ---- keyboard ----
  const held = new Set();
  let shift = false;
  addEventListener('keydown', (e) => {
    shift = e.shiftKey;
    const k = e.code;
    if (k === 'Space') { mm.reset(); e.preventDefault(); }
    else if (k === 'KeyB') mm.triggerPick();
    else if (k === 'KeyN') mm.triggerPlace();
    else if (k === 'KeyT') gizmo.visible = !gizmo.visible;
    else held.add(k);
    if (k.startsWith('Arrow')) e.preventDefault();
  });
  addEventListener('keyup', (e) => { shift = e.shiftKey; held.delete(e.code); });
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  function command() {
    const d = new THREE.Vector3(); camera.getWorldDirection(d);
    let fx = d.x, fy = d.y; const fn = Math.hypot(fx, fy) || 1; fx /= fn; fy /= fn;
    const rx = fy, ry = -fx;
    const fwd = [fx, fy, 0], right = [rx, ry, 0];
    const acc = (v, s) => [v[0] + s[0], v[1] + s[1], 0];
    let move = [0, 0, 0], face = [0, 0, 0];
    if (held.has('KeyW')) move = acc(move, fwd);
    if (held.has('KeyS')) move = acc(move, [-fwd[0], -fwd[1], 0]);
    if (held.has('KeyD')) move = acc(move, right);
    if (held.has('KeyA')) move = acc(move, [-right[0], -right[1], 0]);
    if (held.has('ArrowUp')) face = acc(face, fwd);
    if (held.has('ArrowDown')) face = acc(face, [-fwd[0], -fwd[1], 0]);
    if (held.has('ArrowRight')) face = acc(face, right);
    if (held.has('ArrowLeft')) face = acc(face, [-right[0], -right[1], 0]);
    const top = mm.MAX_SPEED * (shift ? mm.WALK_SCALE : 1);
    const mN = Math.hypot(move[0], move[1]);
    if (mN > 1e-6) { const s = top / mN; move = [move[0] * s, move[1] * s, 0]; }
    else move = [0, 0, 0];
    const fN = Math.hypot(face[0], face[1]);
    face = fN > 1e-6 ? [face[0] / fN, face[1] / fN, 0] : [0, 0, 0];
    return { move, face, speed: mN > 1e-6 ? top : 0 };
  }

  // ---- place the body mesh-groups from FK ----
  function place(qpos) {
    const { wp, wq } = fk(bodies, qpos);
    for (let i = 0; i < bodies.length; i++) {
      bodyGroups[i].position.set(wp[i][0], wp[i][1], wp[i][2]);
      bodyGroups[i].quaternion.set(wq[i][1], wq[i][2], wq[i][3], wq[i][0]);
    }
  }

  const vP = new THREE.Vector3(), vC = new THREE.Vector3(), vMid = new THREE.Vector3(), vDir = new THREE.Vector3();
  function drawGizmo() {
    for (let k = 0; k < 3; k++) {
      const p = mm.Tpos[k], dir = mm.Tdir[k];
      gizSph[k].position.set(p[0], p[1], 0.05);
      const tip = [p[0] + 0.3 * dir[0], p[1] + 0.3 * dir[1], 0.05];
      vP.set(p[0], p[1], 0.05); vC.set(tip[0], tip[1], 0.05);
      vMid.addVectors(vP, vC).multiplyScalar(0.5); gizStk[k].position.copy(vMid);
      vDir.subVectors(vC, vP).normalize(); gizStk[k].quaternion.setFromUnitVectors(_yAxis, vDir);
      gizStk[k].scale.set(1, vP.distanceTo(vC), 1);
    }
    const moving = mm.stateName() === 'MOVE-TO-PICK';
    routeLine.visible = moving;
    cmdArrow.visible = moving;
    if (moving) {
      const pts = mm.routePts;
      const n = Math.min(pts.length, 16);
      for (let i = 0; i < 16; i++) {
        const p = pts[Math.min(i, pts.length - 1)];
        routeArr[3 * i] = p[0]; routeArr[3 * i + 1] = p[1]; routeArr[3 * i + 2] = 0.05;
      }
      routeGeo.attributes.position.needsUpdate = true;
      routeGeo.setDrawRange(0, n);
      const v = mm.cmdVel;
      const vn = Math.hypot(v[0], v[1]);
      if (vn > 1e-3) {
        vP.set(mm.rootPos[0], mm.rootPos[1], 0.05);
        vC.set(mm.rootPos[0] + 0.5 * v[0], mm.rootPos[1] + 0.5 * v[1], 0.05);
        vMid.addVectors(vP, vC).multiplyScalar(0.5); cmdArrow.position.copy(vMid);
        vDir.subVectors(vC, vP).normalize(); cmdArrow.quaternion.setFromUnitVectors(_yAxis, vDir);
        cmdArrow.scale.set(1, vP.distanceTo(vC), 1);
      } else cmdArrow.visible = false;
    }
  }

  // ---- fixed-timestep loop with render interpolation ----
  const DT = mm.DT;
  const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion(), _qi = new THREE.Quaternion();
  const _rq = new Float64Array(36);
  function interp(a, b, t) {
    for (let i = 0; i < 3; i++) _rq[i] = a[i] + (b[i] - a[i]) * t;
    _q0.set(a[4], a[5], a[6], a[3]); _q1.set(b[4], b[5], b[6], b[3]);
    _qi.slerpQuaternions(_q0, _q1, t);
    _rq[3] = _qi.w; _rq[4] = _qi.x; _rq[5] = _qi.y; _rq[6] = _qi.z;
    for (let i = 7; i < 36; i++) _rq[i] = a[i] + (b[i] - a[i]) * t;
    return _rq;
  }

  let acc = 0, last = performance.now() / 1000, lastSpeed = 0;
  let curQ = mm.step([0, 0, 0], [0, 0, 0]), prevQ = curQ;
  let curV = [mm.vasePos.slice(), mm.vaseQuat.slice()], prevV = curV;
  let fps = 0, fpsN = 0, fpsT = last;
  const _v0 = new THREE.Quaternion(), _v1 = new THREE.Quaternion(), _vi = new THREE.Quaternion();
  function frame() {
    const now = performance.now() / 1000;
    acc += Math.min(now - last, 0.1); last = now;
    while (acc >= DT) {
      const c = command(); lastSpeed = c.speed;
      prevQ = curQ; curQ = mm.step(c.move, c.face);
      prevV = curV; curV = [mm.vasePos.slice(), mm.vaseQuat.slice()];
      acc -= DT;
    }
    const f = acc / DT;
    place(interp(prevQ, curQ, f));
    vase.position.set(
      prevV[0][0] + (curV[0][0] - prevV[0][0]) * f,
      prevV[0][1] + (curV[0][1] - prevV[0][1]) * f,
      prevV[0][2] + (curV[0][2] - prevV[0][2]) * f);
    _v0.set(prevV[1][1], prevV[1][2], prevV[1][3], prevV[1][0]);
    _v1.set(curV[1][1], curV[1][2], curV[1][3], curV[1][0]);
    _vi.slerpQuaternions(_v0, _v1, f);
    vase.quaternion.copy(_vi);
    marker.visible = !mm.held;
    drawGizmo();

    controls.target.lerp(new THREE.Vector3(_rq[0], _rq[1], 0.8), 0.2);
    controls.update();

    fpsN++;
    if (now - fpsT >= 0.5) { fps = fpsN / (now - fpsT); fpsN = 0; fpsT = now; }

    const state = mm.stateName();
    let head;
    if (state === 'LOCOMOTION') {
      head = lastSpeed > mm.MAX_SPEED * (1 + mm.WALK_SCALE) / 2 ? 'RUN' : (lastSpeed > 1e-3 ? 'WALK' : 'IDLE');
      head += mm.held ? '  bottle in hand  [N: put it back]' : '  [B: pick up the bottle]';
    } else if (state === 'MOVE-TO-PICK') head = `WALKING TO THE SHELF  [${mm.placing ? 'N' : 'B'}: cancel]`;
    else head = mm.placing ? 'PLACING THE BOTTLE' : 'PICKING UP THE BOTTLE';
    const cid = mm._clipOf(mm.cur);
    const fic = mm.cur - mm.starts[cid];
    setHud(`${head}  ${lastSpeed.toFixed(1)} m/s\nclip [${cid}]: ${mm.clipNames[cid]}\nframe ${fic} (global ${mm.cur})\n` +
      `contact: ${mm.held ? 'ON' : 'off'}\n` +
      `\nrender ${fps.toFixed(0)} fps · sim ${(1 / DT).toFixed(0)} Hz\n` +
      `\nWASD move · arrows face · Shift walk\nB pick · Space reset · T gizmo · drag/scroll camera`);

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  setHud('');
  frame();
}

boot().catch((e) => setHud('error: ' + e.message));

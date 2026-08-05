// Node half of the parity test: runs the JS matcher with the same scripted
// player as test_parity.py and prints all rows as JSON on stdout.
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { MotionMatcher, loadDB } from './js/mm.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, 'data');

const meta = JSON.parse(readFileSync(join(DATA, 'mm.json')));
const model = JSON.parse(readFileSync(join(DATA, 'model.json')));
const binBuf = readFileSync(join(DATA, 'mm.bin'));
const bin = binBuf.buffer.slice(binBuf.byteOffset, binBuf.byteOffset + binBuf.byteLength);

const A = loadDB(meta, bin);
const mm = new MotionMatcher(meta, A, model.bodies);

const N_TICKS = 400;
const TRIGGER_TICK = 30;

const rows = [];
for (let tick = 0; tick < N_TICKS; tick++) {
  if (tick === TRIGGER_TICK) mm.triggerPick();
  const vel = mm.held ? [-0.9, 0, 0] : [0, 0, 0];
  const face = mm.held ? [-1, 0, 0] : [0, 0, 0];
  const q = mm.step(vel, face);
  rows.push([...q, ...mm.vasePos, mm.held ? 1 : 0, mm.animFrame]);
}
console.log(JSON.stringify(rows));

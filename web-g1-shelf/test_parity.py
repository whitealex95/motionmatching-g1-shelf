"""Python <-> JS parity test: both matchers run the same scripted player
(press B at tick 30, carry the bottle away once held, press N at tick 250
to put it back) and every tick's qpos, vase pose, held flag and matched
frame are compared.

    python web-g1-shelf/test_parity.py     (needs node on PATH)
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mm-g1-shelf"))

import config as C
C.APPROX_BIAS = 0.0            # exact NN, like the JS brute force

from data import load_library
from controller import MotionMatcher

N_TICKS = 650
TRIGGER_TICK = 30
PLACE_TICK = 250


def commands(tick, held):
    if held:
        return [-0.9, 0.0, 0.0], [-1.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def run_python():
    m = MotionMatcher(load_library())
    rows = []
    for tick in range(N_TICKS):
        if tick == TRIGGER_TICK:
            m.trigger_pick()
        if tick == PLACE_TICK:
            m.trigger_place()
        vel, face = commands(tick, m.held)
        q = m.step(np.array(vel), np.array(face))
        rows.append(list(q) + list(m.vase_pos)
                    + [1.0 if m.held else 0.0, float(m.animFrame)])
    return np.array(rows)


def run_js():
    out = subprocess.run(
        ["node", os.path.join(ROOT, "web-g1-shelf", "test_parity.mjs")],
        capture_output=True, text=True, check=True)
    return np.array(json.loads(out.stdout.splitlines()[-1]))


def main():
    py = run_python()
    js = run_js()
    assert py.shape == js.shape, (py.shape, js.shape)
    dq = np.abs(py[:, :36] - js[:, :36]).max()
    dv = np.abs(py[:, 36:39] - js[:, 36:39]).max()
    held_same = np.array_equal(py[:, 39], js[:, 39])
    frames_same = np.array_equal(py[:, 40], js[:, 40])
    print(f"max |qpos| diff:  {dq:.2e}")
    print(f"max |vase| diff:  {dv:.2e}")
    print(f"held identical:   {held_same}")
    print(f"frames identical: {frames_same}")
    ok = dq < 1e-4 and dv < 1e-4 and held_same and frames_same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

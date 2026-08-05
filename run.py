#!/usr/bin/env python3
"""Interactive motion matching for the Unitree G1 with a shelf and a vase.

    python run.py                # open the viewer
    python run.py --build-only   # build/refresh the motion library cache and exit

Drive the G1 with WASD, walk up to the shelf ahead, and press B to arm the
grab: once the stance matches the clip well enough (query loss below the
threshold) the pick plays by itself, the arm IK lands the palm on the vase,
and the vase sticks to the hand -- then walk away with it.

Controls
  W / A / S / D ........ move, relative to the camera
  Arrow keys ........... face direction, independent of travel
  Shift (hold) ......... walk instead of run
  B .................... arm/disarm the grab; it plays once the match is good
  Space ................ reset robot and vase
  T .................... toggle the command trajectory gizmo
  Left-drag / right-drag / scroll ... orbit / pan / zoom
  Esc .................. quit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'mm-g1-shelf'))

from data import load_library
from controller import MotionMatcher


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true",
                    help="build/refresh the motion library cache and exit")
    args = ap.parse_args()

    print("Loading motion library (first run builds the feature cache)...")
    lib = load_library()
    print(f"  {len(lib['qpos'])} frames, {len(lib['clip_names'])} clips")

    matcher = MotionMatcher(lib)
    n_search = sum(re - rs for rs, re, _ in matcher.loco_trees)
    print(f"  feature DB ready ({n_search} searchable loco frames, "
          f"{len(matcher.pick_enter)} pick entries)")
    if args.build_only:
        print("Build complete. Run `python run.py` to control the G1.")
        return

    from scene import ShelfScene
    from viewer import InteractiveViewer
    scene = ShelfScene()
    print("Opening viewer -- WASD to move, B at the shelf to grab, Esc to quit.")
    InteractiveViewer(scene, matcher).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive motion matching for the Unitree G1 with an IKEA shelf and bottle.

    python run.py                # open the viewer
    python run.py --build-only   # build/refresh the motion library cache and exit

Drive the G1 with WASD and press B: the robot walks to the pick spot by
itself, plays the pick clip, and the bottle snaps onto the right hand at the
grab -- then walk away with it.

Controls
  W / A / S / D ........ move, relative to the camera
  Arrow keys ........... face direction, independent of travel
  Shift (hold) ......... walk instead of run
  B .................... pick up the bottle (walks to the shelf by itself)
  N .................... put the bottle back (only while holding it)
  Space ................ reset robot and bottle
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
    print(f"  feature DB ready ({n_search} searchable loco frames)")
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

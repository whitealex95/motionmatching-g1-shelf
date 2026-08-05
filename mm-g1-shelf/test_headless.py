"""Headless end-to-end test of the motion matching shelf demo.

Simulates a player: walk toward the shelf, press B, let the pick ride play,
then walk away. Passes only if the vase got grabbed and moved with the hand.

    MUJOCO_GL=egl python test_headless.py [--video out.mp4]
"""
import argparse
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config as C
from data import load_library
from controller import MotionMatcher
from scene import ShelfScene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=None, help='write an .mp4 of the test')
    ap.add_argument('--max-seconds', type=float, default=30.0)
    args = ap.parse_args()

    lib = load_library()
    matcher = MotionMatcher(lib)
    scene = ShelfScene()

    writer = renderer = cam = None
    if args.video:
        import imageio
        import mujoco
        renderer = mujoco.Renderer(scene.model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 3.4, 150.0, -18.0
        writer = imageio.get_writer(args.video, fps=C.FPS, codec='libx264',
                                    quality=8, macro_block_size=None)

    vase_xy = matcher.vase_rest[:2]
    stand = vase_xy - np.array([0.47, 0.14])   # the recorded stance offset
    ride_seen = False
    attempts = 0
    walked_off = 0.0
    result = 'timeout'
    for tick in range(int(args.max_seconds * C.FPS)):
        root = matcher.rootPos[:2]
        to_stand = stand - root
        dist = float(np.linalg.norm(to_stand))

        state = matcher.state_name()
        vel = np.zeros(3); face = np.zeros(3)
        if state == 'PICK':
            ride_seen = True
        elif matcher.held:                       # done: carry it away
            vel = np.array([-0.9, 0.0, 0.0])
            face = np.array([-1.0, 0.0, 0.0])
            walked_off += C.DT
            if walked_off > 2.0:
                result = 'carried away'
                break
        elif dist > 0.12:                        # walk up (slow near the shelf)
            speed = float(np.clip(1.5 * dist, 0.3, 1.2))
            vel = np.array([*(to_stand / dist * speed), 0.0])
            face = np.array([1.0, 0.0, 0.0])
        else:                             # arm the grab (re-arm if it aborted)
            face = np.array([1.0, 0.0, 0.0])
            if attempts < 5 and not matcher.pick_armed:
                matcher.trigger_pick()
                if matcher.pick_armed:
                    attempts += 1

        world = matcher.step(vel, face)
        scene.step(world, matcher.vase_pos, matcher.vase_quat, matcher.held)

        if writer is not None:
            cam.lookat[:] = [float(world[0]) * 0.5 + vase_xy[0] * 0.5,
                             float(world[1]) * 0.5, 0.75]
            renderer.update_scene(scene.data, camera=cam)
            writer.append_data(renderer.render())

        if tick % C.FPS == 0:
            vd = float(np.linalg.norm(matcher.vase_pos[:2] - vase_xy))
            print(f"t={tick / C.FPS:5.1f}s state={state:10s} "
                  f"x={root[0]:5.2f} y={root[1]:5.2f} dist={dist:4.2f} "
                  f"held={matcher.held} vase_moved={vd:4.2f}", flush=True)

    if writer is not None:
        writer.close()
        print(f"wrote {args.video}")

    vase_moved = float(np.linalg.norm(matcher.vase_pos[:2] - vase_xy))
    ok = (result == 'carried away' and matcher.held and ride_seen
          and vase_moved > 0.5)
    print(f"\nresult: {result}; ride_seen={ride_seen}, held={matcher.held}, "
          f"vase_moved={vase_moved:.2f} m")
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

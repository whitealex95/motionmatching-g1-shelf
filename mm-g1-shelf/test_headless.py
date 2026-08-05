"""Headless end-to-end test of the motion matching shelf demo.

Simulates a player: press B once at the spawn, let move-to-pick walk to the
shelf and the pick ride play, then walk away. Passes only if the vase got
grabbed and moved with the hand.

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
    move_seen = ride_seen = False
    walked_off = 0.0
    result = 'timeout'
    matcher.trigger_pick()                       # B, once, right at the spawn
    for tick in range(int(args.max_seconds * C.FPS)):
        root = matcher.rootPos[:2]
        state = matcher.state_name()
        vel = np.zeros(3); face = np.zeros(3)
        if state == 'MOVE-TO-PICK':
            move_seen = True
        elif state == 'PICK':
            ride_seen = True
        elif matcher.held:                       # done: carry it away
            vel = np.array([-0.9, 0.0, 0.0])
            face = np.array([-1.0, 0.0, 0.0])
            walked_off += C.DT
            if walked_off > 2.0:
                result = 'carried away'
                break

        world = matcher.step(vel, face)
        scene.step(world, matcher.vase_pos, matcher.vase_quat, matcher.held)

        if writer is not None:
            cam.lookat[:] = [float(world[0]) * 0.5 + vase_xy[0] * 0.5,
                             float(world[1]) * 0.5, 0.75]
            renderer.update_scene(scene.data, camera=cam)
            writer.append_data(renderer.render())

        if tick % C.FPS == 0:
            vd = float(np.linalg.norm(matcher.vase_pos[:2] - vase_xy))
            stance = float(np.linalg.norm(matcher.stance_xy - root))
            print(f"t={tick / C.FPS:5.1f}s state={state:12s} "
                  f"x={root[0]:5.2f} y={root[1]:5.2f} stance={stance:4.2f} "
                  f"held={matcher.held} vase_moved={vd:4.2f}", flush=True)

    if writer is not None:
        writer.close()
        print(f"wrote {args.video}")

    vase_moved = float(np.linalg.norm(matcher.vase_pos[:2] - vase_xy))
    ok = (result == 'carried away' and matcher.held and move_seen
          and ride_seen and vase_moved > 0.5)
    print(f"\nresult: {result}; move_seen={move_seen}, ride_seen={ride_seen}, "
          f"held={matcher.held}, vase_moved={vase_moved:.2f} m")
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

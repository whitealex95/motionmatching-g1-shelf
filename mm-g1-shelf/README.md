# mm-g1-shelf

Part 2: interactive motion matching with the shelf pick.

```bash
python ../run.py           # or: python run.py from the repo root
```

Walk with WASD. Press **B** and the pick runs as a small state machine:

```
LOCOMOTION --B--> MOVE-TO-PICK --arrived--> PICK --clip end--> LOCOMOTION
```

- **MOVE-TO-PICK** is still motion matching, but the walking command is made
  by the controller. It routes along the rail — the line through the
  recorded stance in its heading direction: first walk to a point 0.6 m
  behind the stance, then straight in facing forward, which is plain
  walking the data has. On the way, each step is warped toward the rail by
  a fraction of the real root motion (ramping from `MOVE_WARP_GAIN` to a
  full projection near the stance), so planted feet never slide; on the
  final leg the future taps also bend onto the line and stop at the stance
  (`MOVE_GOAL_DIST`), so the matcher plays a natural stop there. B again
  cancels, and each leg gives up after `MOVE_TIMEOUT`.
- **PICK** plays the clip to the end with no re-matching. There is no root
  correction after the feet stop: the stance offset that remains when the
  contact flag turns on (~10-15 cm) is absorbed by the vase snap — the
  vase jumps to the recorded grip pose on the palm at that moment and
  follows the hand from then on.

With the trajectory gizmo on (T), MOVE-TO-PICK also shows how its command
is made: a green line to the stance, a blue arrow for the commanded
velocity, a yellow tick for the commanded facing, and the red dots for the
future trajectory the springs predict from that command.

Everything is kinematic; there is no physics anywhere. There is also no
path planning: move-to-pick walks straight at the stance, so starting
behind the shelf walks through it.

| file | role |
|---|---|
| `config.py` | paths, qpos layout, motion-matching and pick settings |
| `data.py` | build/load the motion library (LAFAN loco + the pick clip) |
| `features.py` | the locomotion feature database (27-D) |
| `controller.py` | the matcher: loco search, B plays the pick, vase snap |
| `arm_fk.py` | right-palm forward kinematics (keeps the vase on the hand) |
| `scene.py` | the scene: robot qpos + vase mocap pose, `mj_forward` only |
| `viewer.py` | GLFW viewer, keyboard control, HUD |
| `g1_model.py` | FK for features, sagittal mirror |
| `quat.py`, `springs.py` | math (same as the door/box repos) |
| `test_headless.py` | scripted end-to-end test (`MUJOCO_GL=egl`) |

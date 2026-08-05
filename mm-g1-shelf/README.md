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
  by the controller: walk toward the clip's recorded stance (its root pose
  at the pick entry frame), facing the travel direction and then the stance
  heading. Each step is also warped a little sideways toward the stance —
  at most `MOVE_WARP_GAIN` of the real root motion, so planted feet never
  slide. Every step it checks whether the robot is close enough (position
  and heading); B again cancels, and it gives up after `MOVE_TIMEOUT`.
- **PICK** plays the clip to the end with no re-matching. The root offset
  left over from walking is blended away (`MOVE_LOCK_HALFLIFE`) only while
  the body still moves (`LOCK_SPEED_BAND`) — the entry blend — so there is
  no drift once the robot stands; the last ~2 cm are absorbed by the vase
  snap. When the clip's contact flag turns on, the vase snaps onto the
  right palm with the recorded grip pose and follows the hand from then on.

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

# mm-g1-shelf

Part 2: interactive motion matching with the IKEA shelf-and-bottle pick.

```bash
python ../run.py           # or: python run.py from the repo root
```

Walk with WASD. Press **B** and the pick runs as a small state machine:

```
LOCOMOTION --B--> MOVE-TO-PICK --arrived--> PICK --clip end--> LOCOMOTION
```

- **MOVE-TO-PICK** is still motion matching, but the walking command is made
  by the controller. It routes the walk: straight toward a point 0.6 m
  behind the stance, around a rounded corner there (a small arc, so the
  heading turns continuously), then straight in facing forward, which is
  plain walking the data has. The rail aims `MOVE_OVERSHOOT` past the stance, so
  the commanded walk never slows into the slow-walk dead zone; the pick
  starts the moment the robot crosses the stance plane, still walking. On
  the final leg the root is snapped to the rail — the cross-track and yaw
  parts of the matched motion are projected out (`SNAP_HALFLIFE`) — so it
  cannot diverge from the path. The future taps are read straight off the
  route (walk the remaining path at the approach speed and sample the
  horizons), so the query asks for exactly the trajectory we want. B again
  cancels, and each leg gives up after `MOVE_TIMEOUT`.
- **PICK** plays the clip to the end with no re-matching. The bottle welds
  onto the palm as soon as the live grip pose touches it (`WELD_RADIUS`,
  with the clip's contact frame as fallback). The ~1-2 cm entry offset left
  at that moment is inertialized away (`WELD_HALFLIFE`), so the bottle is
  carried off from where it stood instead of jumping to the hand.
- **PLACE** (N, only while holding) is the same state machine and the same
  clip: the robot walks the same route with the bottle and reaches in. The
  weld is undone at the mirror moment — once the clip's carry phase has
  begun and the grip pose comes back over the rest spot (within
  `WELD_RADIUS`, or at its closest return) the bottle lets go, settles on
  the shelf, and the empty hand plays out the rest of the clip.

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
| `controller.py` | the matcher: loco search, B plays the pick, bottle snap |
| `arm_fk.py` | right-palm forward kinematics (keeps the bottle on the hand) |
| `scene.py` | the scene: robot qpos + bottle mocap pose, `mj_forward` only |
| `viewer.py` | GLFW viewer, keyboard control, HUD |
| `g1_model.py` | FK for features, sagittal mirror |
| `quat.py`, `springs.py` | math (same as the door/box repos) |
| `test_headless.py` | scripted end-to-end test (`MUJOCO_GL=egl`) |

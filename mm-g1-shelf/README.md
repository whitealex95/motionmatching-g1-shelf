# mm-g1-shelf

Part 2: interactive motion matching with the shelf pick.

```bash
python ../run.py           # or: python run.py from the repo root
```

Walk with WASD. Press **B** and the pick clip plays where the robot stands:
the matcher cuts into the clip at its last idle frame and the ride plays to
the end, with no re-matching during it. When the clip's contact flag turns
on, the vase snaps onto the right palm — it takes the grip pose recorded at
the clip's own grab frame — and follows the hand from then on, through the
lift, the hold, and back in locomotion. Everything is kinematic; there is
no physics anywhere.

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

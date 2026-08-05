# mm-g1-shelf

Part 2: interactive motion matching with the shelf pick.

```bash
python ../run.py           # or: python run.py from the repo root
```

Walk with WASD, stand near the shelf, press **B** to arm the grab. While
armed, the matcher watches the entry query loss (how well the live stance
matches the clip's idle frames); once it drops below `PICK_ENTER_THRESHOLD`
it cuts into the pick clip at its best idle frame and the ride plays to the
end. Because the playback is relative (the clip is replayed from wherever
the robot stands), two things fix up the grab:

- **Grab gate.** When the ride reaches the reach phase, the rest of it is
  previewed from the live stance and the arm IK is solved for sampled hand
  targets. If the palm cannot get within `GRAB_CHECK_TOL` of every target,
  the skill aborts back to locomotion — step closer and press B again.
- **Arm IK overlay.** During reach and grasp, a DLS IK on the 7 right-arm
  joints pulls the palm onto the recorded world hand trajectory, which ends
  on the vase (the vase stands exactly where the recorded hand stopped).

When the clip's contact flag turns on, the vase locks to the palm with a
fixed relative pose and follows the hand from then on — through the lift,
the hold, and back in locomotion. Everything is kinematic; there is no
physics anywhere.

| file | role |
|---|---|
| `config.py` | paths, qpos layout, motion-matching and pick settings |
| `data.py` | build/load the motion library (LAFAN loco + the pick clip) |
| `features.py` | feature databases: loco (27-D) and pick (33-D) |
| `controller.py` | the matcher: loco search, B trigger, gate, IK, attachment |
| `arm_ik.py` | DLS IK on the right arm, body-tree FK |
| `shelf.py` | pick-clip entry frames |
| `scene.py` | the scene: robot qpos + vase mocap pose, `mj_forward` only |
| `viewer.py` | GLFW viewer, keyboard control, HUD |
| `g1_model.py` | FK for features, sagittal mirror |
| `quat.py`, `springs.py` | math (same as the door/box repos) |
| `test_headless.py` | scripted end-to-end test (`MUJOCO_GL=egl`) |

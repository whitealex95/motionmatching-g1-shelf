# motionmatching-g1-shelf

A Unitree G1 picks a vase from a shelf — from a single kimodo motion clip to
interactive motion matching, in MuJoCo. Everything is kinematic: the robot
plays back motion, and the vase sticks to the right hand once the grab
happens.

The project has two parts:

| part | folder | what it does |
|---|---|---|
| 1. Scene from one clip | `datagen-g1-shelf/` | Read the kimodo pick-up motion, find where the right hand stops, put a shelf and a vase there, and render the clip. |
| 2. Motion matching | `mm-g1-shelf/` | Interactive motion matching (WASD): press **B** and the robot walks to the shelf and grabs the vase. |
| 3. Web demo | `web-g1-shelf/` | Browser (three.js) port of the demo — the JS matcher reproduces the Python one to float32 precision. Deployed to GitHub Pages. |

```bash
python run.py        # interactive motion matching with the shelf
```

## Layout

```
run.py               entry point for the motion matching demo (part 2)
datagen-g1-shelf/    part 1 — clip conversion + scene + render (see its README)
mm-g1-shelf/         part 2 — motion matching (see its README)
web-g1-shelf/        part 3 — in-browser demo (see its README)
assets/              robot model (menagerie G1)
data/                gmr_lafan1_g1/ locomotion, g1_shelf/ the pick clip + scene meta
tools/               shelf_model.py (the shared shelf + vase scene)
```

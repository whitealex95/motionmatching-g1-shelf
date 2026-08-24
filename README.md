# motionmatching-g1-shelf

A Unitree G1 picks an IKEA bottle from a BILLY shelf — from a single kimodo motion clip to
interactive motion matching, in MuJoCo. Everything is kinematic: the robot
plays back motion, and the bottle sticks to the right hand once the grab
happens.

The project has two parts:

| part | folder | what it does |
|---|---|---|
| 1. Scene from one clip | `datagen-g1-shelf/` | Read the kimodo pick-up motion, find where the right hand stops, put a shelf and bottle there, and render the clip. |
| 2. Motion matching | `mm-g1-shelf/` | Interactive motion matching (WASD): press **B** and the robot walks to the shelf and grabs the bottle. |
| 3. Web demo | `web-g1-shelf/` | Browser (three.js) port of the demo — the JS matcher reproduces the Python one to float32 precision. Deployed to GitHub Pages. |

```bash
python run.py        # interactive motion matching with the IKEA shelf
```

## IKEA assets

`run.py` uses the brown-walnut BILLY shelf asset and black UNDERSÖKA insulated steel flask from
`assets/ikea/`.  MuJoCo cannot load their GLB source files directly; the
converted OBJ/PNG files live in `assets/ikea/mujoco/`.  Regenerate them after
replacing either source asset with:

```bash
BLENDER=/path/to/blender python tools/convert_ikea_assets.py
```

The active pick motion is `data/g1_shelf/pick2.npz`. It is a 34-joint Y-up
G1 representation, so the motion library deterministically retargets it to
MuJoCo qpos as `data/g1_shelf/pick2_g1.npz` when you run `python run.py`.

## Layout

```
run.py               entry point for the motion matching demo (part 2)
datagen-g1-shelf/    part 1 — clip conversion + scene + render (see its README)
mm-g1-shelf/         part 2 — motion matching (see its README)
web-g1-shelf/        part 3 — in-browser demo (see its README)
assets/              robot model and IKEA source/converted assets
data/                gmr_lafan1_g1/ locomotion, g1_shelf/ the pick clip + scene meta
tools/               shelf_model.py (the shared IKEA shelf + bottle scene)
```

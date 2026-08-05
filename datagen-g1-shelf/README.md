# datagen-g1-shelf

Part 1: build the shelf scene from one kimodo motion clip and render it.

The kimodo clip (`output_shelf.npz`) is a G1 standing in place, reaching out
with the right hand, and pulling back. `make_clip.py` finds the frame where
the reaching hand stops — that is the grab — and puts the vase there, with a
shelf under it. From that frame on the contact flag is on and the vase keeps
a fixed pose relative to the palm.

```bash
# 1. Convert the kimodo motion (needs the kimodo package in the env)
python make_clip.py ~/Projects/kimodo/output_shelf.npz
#    -> data/g1_shelf/pick.npz   qpos (T,36), phase, contact, fps
#    -> data/g1_shelf/meta.json  vase position, shelf size, grasp frame

# 2. Render the clip (needs imageio + imageio-ffmpeg)
MUJOCO_GL=egl python render_clip.py     # -> out/pick.mp4
```

Phases: `idle` (standing), `reach` (hand moves out), `grasp` (hand stopped
at the vase), `lift` (vase up and back), `hold` (still, vase in hand).

The shelf and the vase live in `tools/shelf_model.py`, shared with the
motion matching demo, so both parts see the same scene.

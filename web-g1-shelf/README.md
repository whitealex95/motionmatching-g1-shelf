# web-g1-shelf

The shelf-pick demo in the browser (three.js). The JS matcher (`js/mm.js`)
is a port of `mm-g1-shelf/controller.py` and reproduces it to float32
precision: same locomotion search, same move-to-pick route and rail snap,
same pick playback and vase snap.

```bash
# 1. Export the database (mujoco env, from the repo root)
python web-g1-shelf/export_web_data.py
#    -> data/: model.json, mesh.json/.bin, shelf.json, mm.json/.bin (~20 MB)

# 2. Check Python <-> JS parity (needs node)
python web-g1-shelf/test_parity.py

# 3. Serve locally
cd web-g1-shelf && python -m http.server 8000
#    open http://localhost:8000
```

Controls: WASD move, arrows face, Shift walk, **B** pick up the vase
(walks to the shelf by itself; B again cancels), Space reset, T gizmo.

| file | role |
|---|---|
| `export_web_data.py` | dumps the motion DB, meshes, body tree, and shelf scene |
| `js/mm.js` | the matcher port (loco search, move-to-pick, pick, vase snap) |
| `js/fk.js` | body-tree forward kinematics (verified against MuJoCo) |
| `js/quat.js` | wxyz quaternion math |
| `js/main.js` | three.js scene, input, HUD, fixed 30 Hz loop |
| `test_parity.py` / `.mjs` | tick-for-tick Python vs JS comparison |

Deployed by `.github/workflows/pages.yml` on every push to `main` that
touches `web-g1-shelf/`.

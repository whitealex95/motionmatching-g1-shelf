"""The IKEA BILLY shelf and UNDERSÖKA flask used by the Python demo.

The recorded motion is authored around the target in
``data/g1_shelf/meta.json``.  The bottle remains a mocap body: it has no
joints or physics, and the controller places it on the shelf or keeps it
stuck to the right hand after the grab.

MuJoCo does not load GLB files directly.  The OBJ/PNG files used here are
generated from the source GLBs by ``tools/convert_ikea_assets.py``.
"""
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
IKEA_DIR = ROOT / "assets" / "ikea"
# Source GLBs (the web demo loads these directly with three.js).
BILLY_GLB = IKEA_DIR / "BILLY bookcase - brown walnut effect.glb"
BOTTLE_GLB = IKEA_DIR / "UNDERSÖKA insulated steel flask - black.glb"
# MuJoCo conversions (see tools/convert_ikea_assets.py).
BILLY_MESH = IKEA_DIR / "mujoco" / "billy_brown_walnut.obj"
BILLY_TEXTURE = IKEA_DIR / "mujoco" / "billy_brown_walnut_basecolor.png"
BOTTLE_MESH = IKEA_DIR / "mujoco" / "undersoka_flask_black.obj"
BOTTLE_TEXTURE = IKEA_DIR / "mujoco" / "undersoka_flask_black_basecolor.png"

# The converted walnut BILLY shelf is 0.798 x 0.279 x 2.049 m and has a shelf
# top at approximately 0.753 m.  Align this height to the configured bottle
# target so the recorded right-hand trajectory reaches it.  The body origin is
# placed beneath that shelf level, yielding a natural floor contact while
# using a real BILLY shelf rather than a floating bottle.
BILLY_BOTTLE_SHELF_Z = 0.753


def _require_asset(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing MuJoCo asset: {path}\n"
            "Convert the checked-in IKEA GLBs first:\n"
            "  python tools/convert_ikea_assets.py")
    return str(path)


def _add_textured_mesh(spec, mesh_name, mesh_path, texture_name, texture_path,
                       material_name, metallic, roughness):
    """Register one converted OBJ mesh and its base-color PNG material."""
    spec.add_mesh(name=mesh_name, file=_require_asset(mesh_path))
    texture = spec.add_texture(name=texture_name,
                               file=_require_asset(texture_path))
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    material = spec.add_material(name=material_name)
    # The first MjSpec texture slot is the user slot; ``texture=...`` in
    # MJCF maps to the RGB role at index 1.
    material.textures = ("", texture_name)
    material.metallic = metallic
    material.roughness = roughness
    return mesh_name, material_name


def build_model(scene_xml_path, meta, origin_xy=(0.0, 0.0), off_w=1280, off_h=720):
    """Add the IKEA shelf and bottle to the G1 scene and compile it.

    ``meta`` supplies the bottle target from the recorded pick motion;
    ``origin_xy`` shifts that recorded scene away from the robot spawn.
    Returns ``(model, ids)``."""
    spec = mujoco.MjSpec.from_file(scene_xml_path)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    wrist = spec.body("right_wrist_yaw_link")
    wrist.add_site(name="right_palm", pos=[0.10, -0.007, 0.0], size=[0.012] * 3,
                   rgba=[0.1, 0.9, 0.1, 0.35])

    billy_mesh, billy_material = _add_textured_mesh(
        spec, "billy_mesh", BILLY_MESH, "billy_basecolor", BILLY_TEXTURE,
        "billy_material", metallic=0.0, roughness=0.72)
    bottle_mesh, bottle_material = _add_textured_mesh(
        spec, "bottle_mesh", BOTTLE_MESH, "bottle_basecolor", BOTTLE_TEXTURE,
        "bottle_material", metallic=0.85, roughness=0.32)

    # The converted meshes use X=width, Y=depth, Z=up.  A -90 degree yaw puts
    # the BILLY width across the old shelf's Y axis and turns its open front
    # toward the robot at the world origin.
    vx, vy, vz = np.asarray(meta["vase_pos"], dtype=float)
    vx += ox
    vy += oy
    w = spec.worldbody
    billy = w.add_body(name="billy_bookcase",
                       pos=[vx, vy, vz - BILLY_BOTTLE_SHELF_Z],
                       quat=[np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)])
    billy.add_geom(name="billy_bookcase_visual",
                   type=mujoco.mjtGeom.mjGEOM_MESH, meshname=billy_mesh,
                   material=billy_material, rgba=[1.0, 1.0, 1.0, 1.0],
                   contype=0, conaffinity=0)

    # Keep the historical body/ID names: controller and viewer code use them
    # for the kinematic object pose, independent of the visual mesh.
    vase = w.add_body(name="vase", mocap=True, pos=[vx, vy, vz])
    vase.add_geom(name="bottle_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname=bottle_mesh, material=bottle_material,
                  rgba=[1.0, 1.0, 1.0, 1.0], contype=0, conaffinity=0)

    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, off_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, off_h)

    ids = dict(
        vase_mocap=model.body("vase").mocapid[0],
        vase_geoms=[model.geom("bottle_visual").id],
        palm_site=model.site("right_palm").id,
    )
    return model, ids


def set_vase_color(model, ids, held):
    """Keep the textured bottle's actual material in both interaction states."""
    for gid in ids["vase_geoms"]:
        model.geom_rgba[gid] = [1.0, 1.0, 1.0, 1.0]

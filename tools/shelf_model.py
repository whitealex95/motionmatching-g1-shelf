"""The shelf and the vase, shared by every part of this project.

The scene is built in code from the numbers in data/g1_shelf/meta.json,
which datagen computed from the motion clip: the vase stands where the
right hand stops, and the shelf is sized around it.

The vase is a mocap body: it has no joints and no physics. Whoever plays
the motion also sets the vase pose every frame (on the shelf, or stuck to
the hand after the grab).
"""
import numpy as np
import mujoco

BOARD_T = 0.02          # board and panel thickness
BOARD_GAP = 0.28        # vertical distance between boards
N_BOARDS = 3

_WOOD = [0.55, 0.36, 0.20, 1.0]
_WOOD_DARK = [0.38, 0.25, 0.14, 1.0]
_VASE = [0.26, 0.42, 0.62, 1.0]
_VASE_HELD = [0.36, 0.58, 0.82, 1.0]

# Vase shape, relative to its base center: a body, a neck, and a lip.
# The hand grabs the neck, so the grasp point is GRASP_H above the base.
VASE_R = 0.05
GRASP_H = 0.16
VASE_GEOMS = [   # (name, pos_z, radius, half_height)
    ("vase_body", 0.06, 0.050, 0.060),
    ("vase_neck", 0.16, 0.022, 0.040),
    ("vase_lip",  0.215, 0.032, 0.015),
]


def build_model(scene_xml_path, meta, origin_xy=(0.0, 0.0), off_w=1280, off_h=720):
    """Add the shelf and the vase to the G1 scene and compile it.

    `meta` is the "shelf" part of data/g1_shelf/meta.json. `origin_xy`
    moves the whole shelf scene, so the demo can place it away from the
    robot spawn. Returns (model, ids)."""
    spec = mujoco.MjSpec.from_file(scene_xml_path)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    wrist = spec.body("right_wrist_yaw_link")
    wrist.add_site(name="right_palm", pos=[0.10, -0.007, 0.0], size=[0.012] * 3,
                   rgba=[0.1, 0.9, 0.1, 0.35])

    w = spec.worldbody
    top = float(meta["shelf_top_z"])
    front = float(meta["shelf_front_x"]) + ox
    cy = float(meta["shelf_center_y"]) + oy
    depth = float(meta["shelf_depth"])
    width = float(meta["shelf_width"])
    cx = front + depth / 2.0

    shelf = w.add_body(name="shelf", pos=[cx, cy, 0.0])
    for k in range(N_BOARDS):
        z = top - k * BOARD_GAP
        shelf.add_geom(name=f"board_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[0.0, 0.0, z - BOARD_T / 2],
                       size=[depth / 2, width / 2, BOARD_T / 2], rgba=_WOOD)
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        shelf.add_geom(name=f"side_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[0.0, sgn * (width / 2 - BOARD_T / 2), top / 2],
                       size=[depth / 2, BOARD_T / 2, top / 2], rgba=_WOOD_DARK)
    shelf.add_geom(name="back", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[depth / 2 - BOARD_T / 2, 0.0, top / 2],
                   size=[BOARD_T / 2, width / 2, top / 2], rgba=_WOOD_DARK)

    vx, vy, vz = meta["vase_pos"]
    vase = w.add_body(name="vase", mocap=True, pos=[vx + ox, vy + oy, vz])
    for name, z, r, hh in VASE_GEOMS:
        vase.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=[0.0, 0.0, z], size=[r, hh, 0.0], rgba=_VASE,
                      contype=0, conaffinity=0)

    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, off_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, off_h)

    ids = dict(
        vase_mocap=model.body("vase").mocapid[0],
        vase_geoms=[model.geom(n).id for n, _, _, _ in VASE_GEOMS],
        palm_site=model.site("right_palm").id,
    )
    return model, ids


def set_vase_color(model, ids, held):
    """Tint the vase when it is held, as a visible contact label."""
    rgba = _VASE_HELD if held else _VASE
    for gid in ids["vase_geoms"]:
        model.geom_rgba[gid] = rgba

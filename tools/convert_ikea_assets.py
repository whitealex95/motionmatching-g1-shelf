#!/usr/bin/env python3
"""Convert the checked-in IKEA GLBs to Z-up OBJ/PNG assets for MuJoCo.

MuJoCo 3.3 does not read GLB files.  This wrapper invokes Blender's glTF
importer and OBJ exporter, bakes Blender's Z-up coordinate conversion into
the mesh, centres the object in X/Y, and puts its base at Z=0.  The output
therefore works directly with the mocap pose used by the shelf-pick demo.

Usage:
    BLENDER=/path/to/blender python tools/convert_ikea_assets.py

If ``blender`` is on PATH the BLENDER variable is unnecessary.  The output
is deliberately generated alongside the source assets and is ignored when
the source asset is absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IKEA_DIR = ROOT / "assets" / "ikea"
OUTPUT_DIR = IKEA_DIR / "mujoco"
ASSETS = (
    ("BILLY bookcase - brown walnut effect.glb", "billy_brown_walnut"),
    ("UNDERSÖKA insulated steel flask - black.glb", "undersoka_flask_black"),
)


def _worker_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    return parser.parse_args(argv)


def convert_in_blender(argv: list[str]) -> None:
    """Run inside Blender; keep bpy imports out of the normal Python path."""
    import bpy
    from mathutils import Vector

    args = _worker_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {args.source}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    # These IKEA GLBs retain glTF's Y-up coordinates after import.  Bake the
    # change to MuJoCo's Z-up convention into vertex data, so the runtime
    # object pose can remain in the demo's ordinary world frame.
    obj.rotation_euler[0] += 1.5707963267948966
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # MuJoCo poses the bottle by its base centre.  Applying this convention to
    # both meshes makes their placement independent of an importer-specific
    # GLB node origin.
    coords = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    lower = Vector((min(v.x for v in coords), min(v.y for v in coords),
                    min(v.z for v in coords)))
    upper = Vector((max(v.x for v in coords), max(v.y for v in coords),
                    max(v.z for v in coords)))
    offset = Vector((-(lower.x + upper.x) * 0.5,
                     -(lower.y + upper.y) * 0.5,
                     -lower.z))
    for vert in obj.data.vertices:
        vert.co += offset
    obj.location = (0.0, 0.0, 0.0)

    # The GLBs embed a Basecolor WebP.  MuJoCo supports PNG textures, so
    # extract exactly that image rather than depending on an MTL sidecar.
    image = next((im for im in bpy.data.images if "basecolor" in im.name.lower()),
                 None)
    if image is None:
        raise RuntimeError(f"No base-color texture found in {args.source}")
    args.texture.parent.mkdir(parents=True, exist_ok=True)
    # ``image.save()`` preserves the embedded WebP encoder despite a .png
    # suffix.  Copying pixels into a new image forces Blender to encode PNG.
    png = bpy.data.images.new("mujoco_basecolor", width=image.size[0],
                             height=image.size[1], alpha=True)
    png.pixels.foreach_set(image.pixels[:])
    png.filepath_raw = str(args.texture)
    png.file_format = "PNG"
    png.save()

    args.mesh.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(filepath=str(args.mesh), export_selected_objects=True,
                          export_materials=False, export_uv=True,
                          export_normals=True, export_triangulated_mesh=True,
                          forward_axis="Y", up_axis="Z")


def find_blender(requested: str | None) -> str:
    blender = requested or shutil.which("blender")
    if not blender:
        raise SystemExit(
            "Blender is required to convert the GLBs. Install it or run with "
            "BLENDER=/path/to/blender.")
    return blender


def main() -> None:
    if "--blender-worker" in sys.argv:
        index = sys.argv.index("--blender-worker")
        convert_in_blender(sys.argv[index + 1:])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", help="path to the Blender executable")
    args = parser.parse_args()
    blender = find_blender(args.blender)

    for source_name, stem in ASSETS:
        source = IKEA_DIR / source_name
        if not source.is_file():
            raise SystemExit(f"Missing source asset: {source}")
        mesh = OUTPUT_DIR / f"{stem}.obj"
        texture = OUTPUT_DIR / f"{stem}_basecolor.png"
        command = [blender, "--background", "--python", str(Path(__file__).resolve()),
                   "--", "--blender-worker", "--source", str(source),
                   "--mesh", str(mesh), "--texture", str(texture)]
        print(f"Converting {source.name} -> {mesh.relative_to(ROOT)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

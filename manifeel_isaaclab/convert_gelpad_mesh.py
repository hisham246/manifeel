"""Converts the GelSight elastomer gelpad OBJ to USD as a purely visual mesh -- no
collision schema applied at all.

scripts/tools/convert_mesh.py (Isaac Lab's own CLI tool) always constructs a
UsdPhysicsCollisionCfg for the mesh converter, just toggling collision_enabled based on
--collision-approximation -- it never passes collision_props=None, so even
--collision-approximation none leaves the CollisionAPI schema applied (merely
disabled). usb_assembly_env_cfg.py's right_tactile_sensor needs a mesh with NO
CollisionAPI at all: isaaclab_contrib's VisuoTactileSensor._generate_tactile_points()
looks for the elastomer's "visual mesh" via `prim.IsA(UsdGeom.Mesh) and not
prim.HasAPI(UsdPhysics.CollisionAPI)` -- schema presence, not the enabled flag -- so a
disabled-but-still-applied collider still fails that check (confirmed: "No visual mesh
found under elastomer" even after converting with --collision-approximation none).

This calls the same MeshConverter/MeshConverterCfg API convert_mesh.py uses, just with
collision_props=None (skips schema application entirely) instead of an enabled=False
instance of it -- the same collision_props=None convention already used for this
project's other visual-only assets (e.g. usb_assembly_env_cfg.py's own table/light,
inherited from AssemblySceneCfg).

Usage (inside the manifeel-isaaclab Docker container):
    uv run isaaclab -p /workspace/manifeel/manifeel_isaaclab/convert_gelpad_mesh.py
"""

import pathlib

from isaaclab.app import add_launcher_args, launch_simulation

import argparse

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
# The mesh converter uses the Kit asset converter extension (omni.kit.commands etc.),
# which only gets loaded under this physics backend -- ported from convert_mesh.py's
# identical line/comment; omitting this causes "ModuleNotFoundError: No module named
# 'omni.kit.commands'" when importing MeshConverter below.
args_cli.physics = "isaacsim_physx"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_OBJ = str(
    _REPO_ROOT / "assets" / "tacsl" / "mesh" / "gelsight_r15_meshes"
    / "gelsight_r15_finger_gelpad_front_only_rot_z.obj"
)
OUTPUT_USD = str(_REPO_ROOT / "assets" / "tacsl" / "usd" / "gelsight_r15_finger_gelpad.usd")

from isaaclab.physics import PhysicsCfg  # noqa: E402

with launch_simulation(PhysicsCfg(), args_cli):
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    cfg = MeshConverterCfg(
        asset_path=INPUT_OBJ,
        force_usd_conversion=True,
        usd_dir=str(pathlib.Path(OUTPUT_USD).parent),
        usd_file_name=pathlib.Path(OUTPUT_USD).name,
        make_instanceable=False,
        mass_props=None,
        rigid_props=None,
        collision_props=None,  # <-- the actual fix: no schema applied at all
        mesh_collision_props=None,
    )
    converter = MeshConverter(cfg)
    print(f"Generated USD file (visual-only, no collision schema): {converter.usd_path}")

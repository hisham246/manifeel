"""Read-only diagnostic: dumps the USD prim tree under panda_rightfinger (type, API
schemas applied, and whether each is a UsdGeom.Mesh) so we know, with certainty rather
than guessing, which prim is the finger's own native visual mesh and whether
panda_rightfinger itself carries RigidBodyAPI -- both needed to decide how to attach
the TacSL elastomer gelpad correctly (see usb_assembly_env_cfg.py's right_tactile_sensor/
elastomer_right, currently blocked on VisuoTactileSensor._create_physx_views() requiring
the elastomer prim to itself be a PhysX rigid body).

Makes no changes to the stage. Usage:
    uv run isaaclab -p /workspace/manifeel/manifeel_isaaclab/dump_rightfinger_tree.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402

sys.path.insert(0, "/home/aist/hisham/contact_rich_sim/manifeel")

import manifeel_isaaclab.tasks.usb_insertion  # noqa: E402, F401 -- triggers gym.register
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID = "Manifeel-USB-Insertion-Direct"
env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1)

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym  # noqa: E402
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

    env = gym.make(TASK_ID, cfg=env_cfg)
    stage = env.unwrapped.scene.stage if hasattr(env.unwrapped.scene, "stage") else None
    if stage is None:
        import omni.usd  # noqa: E402

        stage = omni.usd.get_context().get_stage()

    root_path = "/World/envs/env_0/Robot/panda_rightfinger"
    root = stage.GetPrimAtPath(root_path)
    print(f"=== {root_path} ===")
    print(f"type: {root.GetTypeName()}")
    print(f"has RigidBodyAPI: {root.HasAPI(UsdPhysics.RigidBodyAPI)}")
    print(f"has CollisionAPI: {root.HasAPI(UsdPhysics.CollisionAPI)}")
    print(f"is UsdGeom.Mesh: {root.IsA(UsdGeom.Mesh)}")
    print()

    for prim in Usd.PrimRange(root):
        if prim.GetPath().pathString == root_path:
            continue
        depth = prim.GetPath().pathString[len(root_path):].count("/")
        indent = "  " * depth
        is_mesh = prim.IsA(UsdGeom.Mesh)
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        print(
            f"{indent}{prim.GetName()}  [{prim.GetTypeName()}]"
            f"{'  MESH' if is_mesh else ''}"
            f"{'  RigidBodyAPI' if has_rigid else ''}"
            f"{'  CollisionAPI' if has_collision else ''}"
        )

    env.close()

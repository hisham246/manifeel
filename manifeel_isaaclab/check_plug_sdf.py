"""Read-only check: does manifeel_usb_plug_1's prototype already carry a proper
UsdPhysics.MeshCollisionAPI(approximation='sdf') schema? (manifeel_usb_plug's prototype
has no 'sdf' child scope at all; manifeel_usb_plug_1's does -- checking whether that's
actually the real schema our tactile sensor needs, or just a same-named coincidence.)

Usage:
    uv run isaaclab -p /workspace/manifeel/manifeel_isaaclab/check_plug_sdf.py
"""

import argparse
import pathlib

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUG_USD = str(_REPO_ROOT / "assets" / "industreal" / "usd" / "USB_plug.usd" / "USB_plug" / "USB_plug.usda")

from isaaclab.physics import PhysicsCfg  # noqa: E402

with launch_simulation(PhysicsCfg(), args_cli):
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

    stage = Usd.Stage.Open(PLUG_USD, load=Usd.Stage.LoadAll)

    for prim in stage.Traverse():
        if not prim.IsInstance():
            continue
        prototype = prim.GetPrototype()
        print(f"{prim.GetPath()} -> {prototype.GetPath()}")
        for p in Usd.PrimRange(prototype):
            has_coll = p.HasAPI(UsdPhysics.CollisionAPI)
            has_mesh_coll = p.HasAPI(UsdPhysics.MeshCollisionAPI)
            approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if has_mesh_coll else None
            print(
                f"  {p.GetPath()}  [{p.GetTypeName()}]  IsA(Mesh)={p.IsA(UsdGeom.Mesh)}"
                f"  CollisionAPI={has_coll}  MeshCollisionAPI={has_mesh_coll}  approximation={approx!r}"
            )
            for attr in p.GetAttributes():
                print(f"      attr: {attr.GetName()} = {attr.Get()}")

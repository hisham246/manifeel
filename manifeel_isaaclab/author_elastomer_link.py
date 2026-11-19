"""Authors a NEW local copy of the AutoMate franka_mimic.usd asset with an added
elastomer_right rigid-body link, fixed-jointed to panda_rightfinger, carrying the
converted GelSight gelpad mesh as its geometry.

Why this is needed (not addable via a sibling Python AssetBaseCfg, which is what was
tried first): isaaclab_contrib's VisuoTactileSensor._create_physx_views() calls
create_rigid_body_view() directly on the sensor's parent prim path -- it requires that
prim to itself be a genuine PhysX rigid body (an articulation LINK, same as
panda_rightfinger already is), not just static visual geometry. The official TacSL
example's own "elastomer" prim is a real link baked into ITS custom test asset's own
USD by its authors -- franka_mimic.usd (AutoMate's stock Franka asset) has no such link
at all, and Isaac Lab has no supported API to retroactively add an articulation link to
an already-built/remote-hosted USD asset at the Python scene-cfg layer. Editing the raw
USD source (this script) is the only way to add one.

Mirrors the original IsaacGym URDF's elastomer_joint_right exactly (manifeel-isaacgymenvs/
assets/tacsl/urdf/tacsl_franka_gelsight_r15.urdf:367-384): a FixedJoint from
panda_rightfinger to a new elastomer_right link, at IDENTITY relative pose (the URDF's
joint origin is xyz="0 0 0" rpy="0 0 0").

Usage (inside the manifeel-isaaclab Docker container -- needs a live Kit session for
the pxr/USD bindings and to locate franka_mimic.usd's local Nucleus cache copy):
    uv run isaaclab -p /workspace/manifeel/manifeel_isaaclab/author_elastomer_link.py

Run manifeel_isaaclab/convert_gelpad_mesh.py FIRST (this script references its output).

CAVEAT: this is the one piece in this whole chain I cannot verify myself (no live Kit
session here) -- specifically whether PhysX's articulation-building pass correctly
absorbs the new FixedJoint+link as an extra link of the EXISTING Robot articulation
(the standard, textbook way to author fixed sub-links in USD/PhysX, but unverified in
this specific codebase/asset). If the env still fails to build afterward, or the
simulation becomes unstable (robot flies apart, NaN poses), that's the signal this
authoring step needs adjustment -- report back what you see.
"""

import argparse
import pathlib

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"  # needed to load the Kit extensions providing pxr/Usd bindings here

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GELPAD_USD = str(_REPO_ROOT / "assets" / "tacsl" / "usd" / "gelsight_r15_finger_gelpad.usd")
OUTPUT_USD = str(_REPO_ROOT / "assets" / "industreal" / "usd" / "franka_mimic_with_elastomer.usd")

FRANKA_MIMIC_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/"
    "Isaac/IsaacLab/AutoMate/franka_mimic.usd"
)

from isaaclab.physics import PhysicsCfg  # noqa: E402

with launch_simulation(PhysicsCfg(), args_cli):
    from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

    # Isaac Lab caches remote Nucleus assets locally under /tmp on first use (confirmed
    # by this session's own run logs: "Loading local copy of remote asset ... from
    # '/tmp/https/...franka_mimic.usd'"). retrieve_file_path() resolves/downloads it the
    # same way the real env build does, so this points at the exact same source asset.
    local_franka_path = retrieve_file_path(FRANKA_MIMIC_URL)
    print(f"Source Franka asset (local cache): {local_franka_path}")

    stage = Usd.Stage.Open(local_franka_path)

    # Find panda_rightfinger by name rather than assuming a specific absolute path --
    # this file's own internal root structure (above panda_rightfinger) isn't something
    # we've inspected directly.
    rightfinger_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == "panda_rightfinger":
            rightfinger_prim = prim
            break
    if rightfinger_prim is None:
        raise RuntimeError("Could not find a prim named 'panda_rightfinger' in franka_mimic.usd")
    rightfinger_path = rightfinger_prim.GetPath()
    print(f"Found panda_rightfinger at: {rightfinger_path}")

    elastomer_path = rightfinger_path.AppendChild("elastomer_right")
    elastomer_prim = UsdGeom.Xform.Define(stage, elastomer_path).GetPrim()

    # Make it a real PhysX rigid body (not just visual geometry) -- this is the actual
    # fix: create_rigid_body_view() needs this prim to be a genuine rigid body.
    UsdPhysics.RigidBodyAPI.Apply(elastomer_prim)
    mass_api = UsdPhysics.MassAPI.Apply(elastomer_prim)
    # Small nonzero mass -- a real gelpad is a few grams; articulation solvers generally
    # need every link to have positive mass. Negligible next to the rest of the arm.
    mass_api.CreateMassAttr(0.005)

    # Visual/ray-casting geometry: reference the already-converted gelpad USD (visual
    # mesh only, no collision schema -- see convert_gelpad_mesh.py) as a child Xform.
    # No explicit primPath given to AddReference() -> uses the referenced file's own
    # defaultPrim automatically, same mechanism the earlier (now-removed) Python-side
    # AssetBaseCfg used successfully for pure geometry display.
    geometry_prim = stage.DefinePrim(elastomer_path.AppendChild("geometry"))
    geometry_prim.GetReferences().AddReference(GELPAD_USD)

    # Fixed joint welding elastomer_right to panda_rightfinger at identity relative
    # pose -- matches the URDF's elastomer_joint_right exactly (origin rpy="0 0 0"
    # xyz="0 0 0"). PhysX's articulation-building pass is expected to absorb this as an
    # additional (zero-DOF) link of the Robot's existing articulation -- the standard
    # way fixed sub-links (sensor mounts, end-effector attachments) are authored in
    # USD/PhysX -- but this specific absorption is the one thing not verified here.
    joint_path = rightfinger_path.AppendChild("elastomer_fixed_joint")
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([rightfinger_path])
    joint.CreateBody1Rel().SetTargets([elastomer_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    pathlib.Path(OUTPUT_USD).parent.mkdir(parents=True, exist_ok=True)
    # Export (not Flatten): saves this layer's own authored prims/edits as a new file
    # while leaving existing external references (the rest of the Franka's sub-meshes,
    # materials) as unresolved reference paths rather than baking everything into one
    # giant flattened file.
    stage.GetRootLayer().Export(OUTPUT_USD)
    print(f"Wrote modified Franka asset to: {OUTPUT_USD}")

"""Adds SDF-approximation collision to the USB plug's mesh, in place, in the existing
local USD_plug.usd asset.

Why: isaaclab_contrib's VisuoTactileSensor needs this at TWO separate levels, not just
one:
  1. Static discovery -- _find_contact_object_components()'s is_sdf_mesh() check just
     wants UsdPhysics.MeshCollisionAPI with approximation=="sdf" (visuotactile_sensor.py
     :374-379). A plain UsdPhysics.MeshCollisionAPI.Apply() + CreateApproximationAttr
     ("sdf") satisfies this and let env construction succeed.
  2. Runtime SDF queries -- create_sdf_shape_view() (used every step by
     _update_force_field() for penetration-depth queries) additionally needs the real
     PhysX-specific PhysxSDFMeshCollisionAPI schema with actual baked SDF cooking
     parameters (resolution etc.) -- the generic schema from (1) only satisfies static
     validation, not the real GPU tensor view ("Failed to find CollisionAPI and
     PhysxSDFMeshCollisionAPI for prim..." / "did not match any SDF" hit at env.reset()
     despite (1) already being in place).
Both are applied here via isaaclab.sim.schemas.apply_mesh_collision_properties with a
PhysxSDFMeshCfg fragment -- the same tested code path the rest of Isaac Lab's asset
pipeline uses, rather than hand-authoring raw PhysX attribute names via pxr directly.
sdf_resolution=128 matches this asset's own leftover urdf:resolution=128 metadata (see
the separate manifeel_usb_plug_1/sdf Scope -- turned out to be inert metadata from the
original URDF conversion, not a real schema, but it's still the intended resolution).

SDF is a supported PhysX collision approximation for dynamic rigid bodies (unlike plain
triangleMesh/meshSimplification, which PhysX rejects for dynamic bodies -- see the
earlier "falling back to convexHull" warning hit elsewhere in this same session), and
is generally MORE accurate than whatever approximation is there now, so applying/
overwriting it on the plug's mesh should be a quality improvement for existing
grasp/contact-force behavior, not a regression -- not a separate added mesh, the same
collision geometry just gets a (better) approximation type.

Usage (inside the manifeel-isaaclab Docker container):
    uv run isaaclab -p /workspace/manifeel/manifeel_isaaclab/author_plug_sdf_collision.py
"""

import argparse
import pathlib

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"  # needed to load the Kit extensions providing pxr/Usd bindings here

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUG_USD = str(_REPO_ROOT / "assets" / "industreal" / "usd" / "USB_plug.usd" / "USB_plug" / "USB_plug.usda")

from isaaclab.physics import PhysicsCfg  # noqa: E402

with launch_simulation(PhysicsCfg(), args_cli):
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

    from isaaclab.sim.schemas import apply_mesh_collision_properties  # noqa: E402
    from isaaclab_physx.sim.schemas.schemas_cfg import PhysxSDFMeshCfg  # noqa: E402

    stage = Usd.Stage.Open(PLUG_USD, load=Usd.Stage.LoadAll)

    # manifeel_usb_plug (not _1) is the instance contact_object_prim_path_expr actually
    # points at. It's instanceable (--make-instanceable, presumably for efficient
    # cloning across many parallel training envs), which is why it showed up as an
    # empty Xform with no children: USD forbids authoring directly onto a shared
    # instancing prototype ("Cannot create prim spec at path </__Prototype_N/...>;
    # authoring to an instancing prototype is not allowed" -- hit directly on an
    # earlier attempt). The _1 sibling's own prototype has an "sdf" child, but it
    # turned out to be leftover custom metadata (urdf:resolution=128) from the
    # original URDF conversion, not an actual UsdPhysics.MeshCollisionAPI -- so this
    # schema needs to be authored for real, not just pointed at.
    #
    # Fix: temporarily un-instance this one specific prim (SetInstanceable(False)),
    # which exposes its real underlying content for direct editing instead of a shared
    # prototype, make the edit, then leave it un-instanced (see note near the bottom of
    # this script for why restoring True turned out to discard the edit).
    # Not requiring IsInstance() here: a prior run of this script already
    # un-instanced this specific prim and that persisted to disk, so on a second run
    # it's a plain Xform now, not an instance -- only matching by name/type (Xform,
    # not the nested Mesh of the same name one level deeper) keeps this re-runnable.
    target_instance = None
    for prim in stage.Traverse():
        if prim.GetName() == "manifeel_usb_plug" and not prim.IsA(UsdGeom.Mesh):
            target_instance = prim
            break
    if target_instance is None:
        raise RuntimeError(f"Could not find a Xform prim named 'manifeel_usb_plug' in {PLUG_USD}")
    print(f"Found target: {target_instance.GetPath()} (IsInstance={target_instance.IsInstance()})")

    target_instance.SetInstanceable(False)

    mesh_prim = None
    for prim in Usd.PrimRange(target_instance):
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
    if mesh_prim is None:
        target_instance.SetInstanceable(True)  # restore before failing
        raise RuntimeError(f"No Mesh-typed prim found under {target_instance.GetPath()} after un-instancing")
    print(f"Found plug mesh at: {mesh_prim.GetPath()}")

    had_collision_api = mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
    if not had_collision_api:
        UsdPhysics.CollisionAPI.Apply(mesh_prim)

    # apply_mesh_collision_properties applies UsdPhysics.MeshCollisionAPI as the anchor
    # automatically, then dispatches the PhysxSDFMeshCfg fragment -- whose func authors
    # BOTH the real PhysxSDFMeshCollisionAPI cooking schema (needed for the runtime SDF
    # tensor view) AND the shared physics:approximation="sdf" token (needed for the
    # static is_sdf_mesh() discovery check) in one call, via the same tested code path
    # the rest of Isaac Lab's asset pipeline uses -- rather than two separate
    # hand-authored schemas that could drift out of sync with each other.
    # sdf_resolution=128 matches this asset's own leftover urdf:resolution=128 hint.
    success = apply_mesh_collision_properties(
        str(mesh_prim.GetPath()), [PhysxSDFMeshCfg(sdf_resolution=128)], stage=stage
    )
    print(f"apply_mesh_collision_properties on {mesh_prim.GetPath()}: success={success}"
          f" (CollisionAPI was already applied: {had_collision_api})")

    # NOT restoring SetInstanceable(True): tried that first and the readback came back
    # empty -- USD instancing composition discards ANY local override on an instanced
    # subtree once re-enabled (the whole point of instancing is guaranteed-identical
    # shared content), so it silently threw the edit away. Leaving this one prim
    # un-instanced is the actual fix, not just a fallback -- the cost is negligible
    # (one small plug mesh, not the whole robot; the 128-env cloning this asset's
    # comment refers to is Isaac Lab's separate replicate_physics mechanism).

    stage.GetRootLayer().Save()
    print(f"Saved in place: {PLUG_USD}")

    # Readback check -- editing a prototype prim (rather than a regular prim) is the
    # one part of this not verified ahead of time; confirm the edit actually persisted
    # to disk rather than silently no-op'ing (e.g. if it got routed to the wrong layer).
    from pxr import PhysxSchema  # noqa: E402

    reopened = Usd.Stage.Open(PLUG_USD, load=Usd.Stage.LoadAll)
    found_sdf = False
    found_physx_sdf = False
    for prim in reopened.Traverse():
        candidates = [prim]
        if prim.IsInstance():
            candidates = list(Usd.PrimRange(prim.GetPrototype()))
        for cand in candidates:
            if (
                cand.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(cand).GetApproximationAttr().Get() == "sdf"
            ):
                found_sdf = True
                print(f"Readback confirms sdf MeshCollisionAPI on: {cand.GetPath()}")
            if cand.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                found_physx_sdf = True
                print(f"Readback confirms PhysxSDFMeshCollisionAPI on: {cand.GetPath()}")
    if not found_physx_sdf:
        print("WARNING: readback did NOT find PhysxSDFMeshCollisionAPI -- the runtime SDF view will still fail.")
    if not found_sdf:
        print("WARNING: readback did NOT find the sdf schema after saving -- the edit may not have persisted.")

"""USB plug/socket insertion task config for manifeel.

Subclasses Isaac Lab's AutoMate AssemblyTask, pointing at the USB-specific USD
assets converted from manifeel-isaacgymenvs' URDF/OBJ sources (see
scripts/tools/convert_urdf.py), instead of the stock Nucleus-hosted 00015
peg/hole assembly.

Diameter/height/friction values below come from
manifeel/assets/tacsl/yaml/tacsl_asset_info_power.yaml (USB_plug, USB_socket).
Mass is left to PhysX via density (matches the original IsaacGym Preview setup,
which set AssetOptions.density in code rather than an explicit mass).

Reset convention is ported from manifeel-isaacgymenvs' TacSLTaskUSB (not AutoMate's
own disassembly-curriculum reset -- that's designed around AutoMate's own recorded
demonstration data, which we don't have for this asset). See usb_assembly_env.py's
UsbInsertionEnv.randomize_initial_state() override for the actual reset logic; the
assembly_dir/obj_path/fingertip_midpoint_*/plug_*_in_gripper_* fields below are the
config inputs that override reads.
"""

from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.automate.assembly_tasks_cfg import AssemblyTask, FixedAssetCfg, HeldAssetCfg

# Converted via: scripts/tools/convert_urdf.py <manifeel-isaacgymenvs urdf> <this dir>/<name>.usd
USB_USD_DIR = "/home/aist/hisham/contact_rich_sim/manifeel/assets/industreal/usd"

# Real OBJ meshes (same geometry the USD conversion came from) -- used by AssemblyEnv.__init__
# for gripper_open_width (bounding-box-based) and the warp SDF mesh load. Pointing these at our
# actual assets makes that code path correct, not stubbed, even though we don't use the SDF reward.
USB_MESH_DIR = "/home/aist/hisham/contact_rich_sim/manifeel-isaacgymenvs/assets/industreal/mesh/TVB_mesh/"


@configclass
class UsbPlugCfg(HeldAssetCfg):
    # convert_urdf.py's "output" is actually a directory: it nests the real root layer
    # at <output>.usd/<name>/<name>.usda alongside config.yaml/.asset_hash/payloads/Textures.
    usd_path: str = f"{USB_USD_DIR}/USB_plug.usd/USB_plug/USB_plug.usda"
    obj_path: str = "manifeel_usb_plug.obj"
    diameter: float = 0.007986
    # tacsl_asset_info_power.yaml's "length: 0.050" is NOT the mesh's actual bounding-box
    # extent -- confirmed by directly measuring manifeel_usb_plug.obj's vertices (matches
    # the SDF-cooking log's "Extents" seen during the original IsaacGym Preview run too).
    # height must be the real mesh length for the reset-placement math in usb_assembly_env.py
    # (AutoMate's own HeldAssetCfg.height convention is "the asset's physical length", which
    # this matches).
    height: float = 0.120958
    friction: float = 1.0


@configclass
class UsbSocketCfg(FixedAssetCfg):
    usd_path: str = f"{USB_USD_DIR}/USB_socket.usd/USB_socket/USB_socket.usda"
    obj_path: str = "manifeel_usb_socket.obj"
    diameter: float = 0.0081
    # Same correction as UsbPlugCfg.height -- real mesh Z-extent, not the yaml's functional
    # "height: 0.028" value. Matches manifeel_usb_socket.obj's measured bounding box.
    height: float = 0.03726
    base_height: float = 0.0
    friction: float = 0.5


@configclass
class UsbInsertion(AssemblyTask):
    name = "usb_insertion"

    fixed_asset_cfg = UsbSocketCfg()
    held_asset_cfg = UsbPlugCfg()
    asset_size = 8.0
    duration_s = 10.0

    # Real mesh dir, for AssemblyEnv.__init__'s gripper_open_width computation and warp SDF
    # mesh load (both just need a valid OBJ path; see USB_MESH_DIR above).
    assembly_dir: str = USB_MESH_DIR

    # --- Socket (fixed_asset) placement: ported directly from TacSLTaskUSB.yaml's
    # randomize.socket_pos_xyz_initial/_noise/socket_rot_initial/_noise. randomize_fixed_initial_state
    # (reused as-is from AssemblyEnv) adds pos noise + fixed_asset_z_offset to fixed_asset's
    # init_state.pos below, and a yaw-only rotation noise -- socket_rot_noise is all-zero in the
    # original, so orn_range_deg=0 reproduces that (no rotation randomization at all).
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0
    fixed_asset_z_offset: float = 0.0  # z is already absolute in fixed_asset.init_state.pos below

    # --- Gripper target pose before grasping (TacSLTaskUSB.yaml randomize.fingertip_midpoint_*).
    # Consumed by UsbInsertionEnv.randomize_initial_state()'s override, not AssemblyEnv itself.
    fingertip_midpoint_pos_initial: list = [0.5, 0.0, 0.15]
    fingertip_midpoint_pos_noise: list = [0.1, 0.1, 0.05]
    fingertip_midpoint_rot_initial: list = [3.1416, 0.0, 0.0]  # euler, rad
    fingertip_midpoint_rot_noise: list = [0.1, 0.1, 0.1]

    # --- Plug-in-gripper placement (TacSLTaskUSB.yaml env.peg_in_hand_buffer /
    # randomize.plug_pos_z_in_gripper_noise_multiplier / plug_pos_in_gripper_noise_xy /
    # plug_noise_rot_in_gripper). Also consumed by the override above.
    peg_in_hand_buffer: float = 1.0
    plug_pos_z_in_gripper_noise_multiplier: float = 0.0
    plug_pos_in_gripper_noise_xy: list = [0.0, 0.0]
    plug_noise_rot_in_gripper: list = [0.0, 0.1745, 0.0]  # ~10 deg about local y

    # --- Franka default/open/close gripper widths (TacSLTaskUSB.yaml env.franka_*_gripper_width,
    # num_gripper_close_sim_steps). franka_arm_initial_dof_pos is set via UsbInsertionEnvCfg.ctrl
    # .reset_joints instead (that's where AssemblyEnv._reset_idx actually reads it from).
    franka_open_gripper_width: float = 0.08
    franka_close_gripper_width: float = 0.0145
    num_gripper_close_sim_steps: int = 30

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    # Remaining fields the stock Insertion task defines that AssemblyEnv's success-check
    # (automate_algo.check_plug_inserted_in_socket) and other base methods read directly --
    # same defaults as the stock task, not asset-specific.
    engage_height_thresh: float = 0.01
    success_height_thresh: float = 0.003
    close_error_thresh: float = 0.015

    # USB_socket.urdf has zero joints -- it's a plain rigid body, not a real articulation
    # (confirmed via the asset pipeline research; unlike the stock Insertion task's own
    # Nucleus-hosted socket, which apparently does have joint structure and uses
    # ArticulationCfg + fix_root_link=True). Model it as a kinematic RigidObjectCfg instead:
    # kinematic_enabled makes it immovable while still fully collidable, which is the rigid-
    # body equivalent of "fixed_root_link" for an asset with no joints to fix in the first place.
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=[
                UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                PhysxRigidBodyCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
            ],
            mass_props=sim_utils.MassCfg(density=8000.0),  # tacsl_asset_info_power.yaml: USB_socket
            collision_props=PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        # Matches TacSLTaskUSB.yaml randomize.socket_pos_xyz_initial=[0.5, 0.0, 0.025]
        # (noise/rotation applied on top at reset by randomize_fixed_initial_state).
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.025),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # A free rigid body: gravity applies normally. The reset sequence disables gravity
    # GLOBALLY and temporarily (scene.physics_manager, in UsbInsertionEnv's reset override)
    # while the plug is placed and grasped, then restores it -- not a permanent per-body flag.
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassCfg(density=80000.0),  # tacsl_asset_info_power.yaml: USB_plug
            collision_props=PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.15),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

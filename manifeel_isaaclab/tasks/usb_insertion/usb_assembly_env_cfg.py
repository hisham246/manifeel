"""Scene + env config for the USB insertion task, adding a wrist-mounted RGB camera.

Uses the non-Tiled ``Camera`` sensor (via ``CameraCfg``) -- ``TiledCamera`` is known to
hang on RTX 5090 (Blackwell), confirmed earlier this session against the stock AutoMate
visuo-tactile-sensor demo.

Offset/resolution/FOV are ported from the original IsaacGym Preview wrist camera config
(manifeel/manifeel/config/task/TacSLTaskUSB.yaml, the "wrist" entry):
    camera_pose: [[0.045, 0, 0.04], [0.5, 0, 0.866, 0]]   # xyz pos, xyzw quat, rel. panda_hand
    image_size: [256, 256]
    horizontal_fov: 75.0

The rotation convention (ROS vs OpenGL vs world) between IsaacGym Preview's
attach_camera_to_body and Isaac Lab's CameraCfg.OffsetCfg has NOT been visually verified
yet -- flagged for milestone 7 (standalone smoke test with an actual rendered frame).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.automate.assembly_env_cfg import AssemblyEnvCfg, AssemblySceneCfg, CtrlCfg

from .usb_assembly_tasks_cfg import UsbInsertion

# Solve for the aperture that gives horizontal_fov=75deg at Isaac Lab's default focal_length
# (FOV depends only on the aperture/focal_length ratio, so the absolute unit convention
# doesn't matter as long as both numbers use the same one).
_FOCAL_LENGTH = 24.0
_HORIZONTAL_FOV_DEG = 75.0
_HORIZONTAL_APERTURE = 2 * _FOCAL_LENGTH * math.tan(math.radians(_HORIZONTAL_FOV_DEG) / 2)


@configclass
class UsbAssemblySceneCfg(AssemblySceneCfg):
    """AutoMate assembly scene plus a wrist camera on the Franka hand."""

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.045, 0.0, 0.04),
            rot=(0.5, 0.0, 0.866, 0.0),  # xyzw, ported from TacSLTaskUSB.yaml camera_pose
            # IsaacGym Preview's attach_camera_to_body forward axis is +X -- matches Isaac
            # Lab's "world" convention (fwd=+X, up=+Z), not the default "ros" (fwd=+Z, up=-Y).
            convention="world",
        ),
        width=256,
        height=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL_LENGTH,
            horizontal_aperture=_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 10.0),
        ),
    )


@configclass
class UsbInsertionEnvCfg(AssemblyEnvCfg):
    """Env config for the USB plug/socket insertion task."""

    task_name: str = "usb_insertion"
    tasks: dict = {"usb_insertion": UsbInsertion()}

    scene: UsbAssemblySceneCfg = UsbAssemblySceneCfg(num_envs=128, env_spacing=2.0)

    # TacSLTaskUSB.yaml's randomize.franka_arm_initial_dof_pos -- the Franka arm config
    # _reset_idx resets to via _set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints)
    # before UsbInsertionEnv.randomize_initial_state() moves it to the grasp-approach pose.
    ctrl: CtrlCfg = CtrlCfg(
        reset_joints=[0.0002, -0.1591, 0.0005, -2.0966, -0.0002, 1.9335, 0.7855],
    )

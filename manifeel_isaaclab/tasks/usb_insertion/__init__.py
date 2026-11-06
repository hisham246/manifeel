"""Registers the USB plug/socket insertion task with gymnasium.

Mirrors isaaclab_tasks/contrib/automate/__init__.py's registration pattern
exactly. This package lives outside the IsaacLab source tree (manifeel owns
it) -- confirmed earlier this session that gym task discovery only requires
this module to be imported before gym.make()/gym.spec() is called, not that
it live under isaaclab_tasks.
"""

import gymnasium as gym

gym.register(
    id="Manifeel-USB-Insertion-Direct",
    entry_point=f"{__name__}.usb_assembly_env:UsbInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.usb_assembly_env_cfg:UsbInsertionEnvCfg",
    },
)

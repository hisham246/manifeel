from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NullEnvRunner(BaseImageRunner):
    """No-op env runner: skips simulated eval rollouts when no sim is available."""

    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}

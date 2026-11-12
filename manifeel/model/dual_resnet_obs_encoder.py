from typing import Dict

import torch
import torch.nn as nn

from maniflow.model.common.module_attr_mixin import ModuleAttrMixin
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder


class DualResNetObsEncoder(ModuleAttrMixin):
    """Encodes each rgb-type obs key (front, tactile_force_field_right) through its own
    from-scratch ResNet-18, matching the ManiFlow paper's described 2D encoder ("we
    train a ResNet-18 encoder from scratch... adaptive visual features... fed into
    DiT-X for cross-attention") -- applied to TacFF too, per the ManiFeel paper's choice
    to encode TacFF inside a vision encoder, while keeping ManiFlow's actual multi-token
    cross-attention conditioning (not diffusion_policy's single pooled global vector).

    One TimmObsEncoder per rgb key, not one shared instance over the whole shape_meta:
    TimmObsEncoder asserts every rgb-type key shares the same (H,W), which breaks the
    moment you mix front's (256,256) with tactile_force_field_right's (10,14) -- giving
    each key its own instance sidesteps that assertion entirely (each only ever sees one
    key, so there's nothing to conflict with).

    feature_aggregation='avg' (not None): for model_name='resnet18' (plain, not 'r3m'),
    aggregate_feature() with feature_aggregation=None returns an unpooled 3D tensor,
    which forward()'s very next line asserts must be 2D -- confirmed crashes on a CPU
    probe. 'avg' pools it down to 2D first, same effect as a standard resnet's own
    AdaptiveAvgPool2d, verified working via the same probe.

    Token layout: this returns ONE combined vector (all keys/obs_steps concatenated and
    flattened), not a genuine multi-token sequence per modality -- checked directly
    against maniflow.model.diffusion.ditx.DiTX's own forward(): its
    vis_cond_obs_emb = nn.Linear(cond_dim, n_emb) just linearly projects whatever
    (B, L, cond_dim) it's handed; it does not manufacture multiple tokens internally.
    Since cond_dim is set from obs_encoder.output_shape()[-1] -- the encoder's *entire*
    flat output size -- ManiFlowTransformerImagePolicy's own `.reshape(B, -1, Do)` always
    collapses to L=1 for a single encoder instance, in the reference TimmObsEncoder usage
    just as much as here. So this still gets you both things actually asked for --
    TacFF through its own real from-scratch ResNet-18 (not low_dim-flattened), and
    ManiFlow's actual DiTX backbone/training objective -- just not a richer per-modality
    token sequence, because the reference implementation doesn't provide one either.
    """

    def __init__(self, shape_meta: dict, n_obs_steps: int,
                 model_name: str = 'resnet18', pretrained: bool = False, frozen: bool = False,
                 feature_aggregation: str = 'avg', downsample_ratio: int = 32,
                 transforms_by_key: dict = None, use_group_norm: bool = True,
                 imagenet_norm: bool = False):
        """transforms_by_key: {key: [transform, ...]} -- per-key, not applied uniformly,
        since the paper's augmentations (color jitter especially) are meant for real RGB
        photos and are physically meaningless on tactile_force_field_right's normal/shear
        force channels. Keys not present get no augmentation (transforms=None)."""
        super().__init__()

        self.n_obs_steps = n_obs_steps
        transforms_by_key = transforms_by_key or {}

        obs_shape_meta = shape_meta['obs']
        self.rgb_keys = sorted(k for k, a in obs_shape_meta.items() if a.get('type', 'low_dim') == 'rgb')
        self.low_dim_keys = sorted(k for k, a in obs_shape_meta.items() if a.get('type', 'low_dim') == 'low_dim')
        assert len(self.rgb_keys) > 0, "DualResNetObsEncoder needs at least one rgb key"

        self.rgb_encoders = nn.ModuleDict()
        for key in self.rgb_keys:
            attr = obs_shape_meta[key]
            key_shape_meta = {
                'obs': {key: {**attr, 'horizon': n_obs_steps}},
                'action': shape_meta['action'],
            }
            self.rgb_encoders[key] = TimmObsEncoder(
                shape_meta=key_shape_meta,
                model_name=model_name, pretrained=pretrained, frozen=frozen,
                global_pool='', transforms=transforms_by_key.get(key, None), use_group_norm=use_group_norm,
                share_rgb_model=False, imagenet_norm=imagenet_norm,
                feature_aggregation=feature_aggregation, downsample_ratio=downsample_ratio,
                position_encording='sinusoidal',
            )

        # low_dim keys get projected to resnet18's native per-step channel width (512)
        # purely so a single Linear per key is enough -- doesn't need to match anything
        # else, since everything just gets concatenated flat (see forward()).
        feature_dim = 512
        self.low_dim_proj = nn.ModuleDict()
        for key in self.low_dim_keys:
            dim = 1
            for d in obs_shape_meta[key]['shape']:
                dim *= d
            self.low_dim_proj[key] = nn.Linear(dim, feature_dim)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = next(iter(obs_dict.values())).shape[0]
        features = []

        for key in self.rgb_keys:
            features.append(self.rgb_encoders[key]({key: obs_dict[key]}))  # (B, T*512)

        for key in self.low_dim_keys:
            x = obs_dict[key].reshape(B, self.n_obs_steps, -1).to(self.device)  # (B, T, dim)
            features.append(self.low_dim_proj[key](x).reshape(B, -1))  # (B, T*512)

        return torch.cat(features, dim=-1)  # (B, N_keys*T*512) -- one combined vector

    @torch.no_grad()
    def output_shape(self):
        example = {}
        for key in self.rgb_keys:
            shape = tuple(self.rgb_encoders[key].shape_meta['obs'][key]['shape'])  # (C,H,W)
            example[key] = torch.zeros((1, self.n_obs_steps, shape[1], shape[2], shape[0]), device=self.device)
        for key in self.low_dim_keys:
            dim = self.low_dim_proj[key].in_features
            example[key] = torch.zeros((1, self.n_obs_steps, dim), device=self.device)
        out = self.forward(example)
        return out.shape[1:]

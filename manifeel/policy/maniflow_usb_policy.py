from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy


class ManiFlowUsbPolicy(ManiFlowTransformerImagePolicy):
    """manifeel-side subclass of ManiFlow's transformer image policy, for the USB
    insertion task.

    This exists as an extension point separate from the `maniflow` package (installed
    editable as a sibling repo -- see /home/aist/hisham/contact_rich_sim/maniflow),
    mirroring how manifeel's diffusion_policy-based policies (e.g. diffusion_tacff_policy.py)
    subclass diffusion_policy's base classes rather than vendoring its code.

    No overrides yet -- behaves identically to the base class. The planned extension is
    swapping tactile_force_field_right for an MLP-predicted wrench at inference time
    (once that TacFF->wrench mapping is trained), since Isaac Lab deployment only has
    the real ContactSensor wrench available, not the TacSL tactile force field.
    """
    pass

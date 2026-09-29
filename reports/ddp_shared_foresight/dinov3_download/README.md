# Requested DINOv3 artifact

Separate download of **ViT-L/16, LVD-1689M, timm format**. This download does not change the registered FLUX target, student architecture or experiment configuration.

Source: [timm model distribution](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/tree/30c1109559f65dea34316b0d4842d35c5771fe11), immutable revision `30c1109559f65dea34316b0d4842d35c5771fe11`. [DINOv3 upstream](https://github.com/facebookresearch/dinov3) documents timm support. The public distribution includes its model card and DINOv3 license.

This is a converted checkpoint, not the Meta Transformers artifact: timm omits the original all-zero QKV biases and generates FP32 RoPE periods. The check uses timm defaults; no numerical equivalence to another implementation is claimed. Download integrity is checked against the pinned timm SHA256, with strict model loading and a real current training image CPU forward. See `DOWNLOAD.json` and `CPU_LOAD_CHECK.json` for actual outcomes.

Local directory:
`/mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm`

Reproduce download and verification (existing correct files are reused):

```bash
cd /mnt/project/VLA-Drive-ddp-foresight-20260928
/root/miniconda3/envs/ddp/bin/python -m tools.foresight.download_dinov3 \
  --root /mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm
```

Offline inference example in the existing `ddp` environment:

```python
import timm
import torch
from safetensors.torch import load_file
root = "/mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm"
model = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0)
model.load_state_dict(load_file(root + "/model.safetensors"), strict=True)
model.eval().requires_grad_(False)
# Apply input normalization/resize from the downloaded config.json.
# model.forward_features(pixels) returns image feature tokens.
```

No GPU allocation, optimizer updates, auxiliary-target substitution or planning evaluation is required for this artifact check. Large weights and private images stay outside Git.

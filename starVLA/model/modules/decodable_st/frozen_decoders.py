"""Fixed, strictly loaded native heads. Input autograd is deliberately enabled."""
import torch
from torch import nn


class FrozenDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        self.decoder.eval()
        return self


class FrozenViGTDecoder(FrozenDecoder):
    def __init__(self, decoder, contract):
        super().__init__(decoder)
        self.contract = contract.validate()

    def forward(self, context, xyz):
        if tuple(context.shape[1:]) != (self.contract.channels,self.contract.height,self.contract.width):
            raise ValueError("ViGT context shape differs from verified checkpoint")
        valid = self.contract.in_roi(xyz)
        # Reject invalid queries before upstream's internal clamp. The caller may
        # separately retain an OOR flag in its evaluation denominator.
        if not bool(valid.all()):
            raise ValueError("ViGT query outside the verified ROI or nonfinite")
        return self.decoder(context, xyz)


def sgdrive_logits_layout(raw, height_bins=16):
    b, c, x, y = raw.shape
    if c != 2*height_bins:
        raise ValueError("SGDrive native channel/height contract mismatch")
    # Exact native: [B,32,X,Y] -> [B,2,Z,X,Y] -> [B,2,X,Y,Z].
    return raw.reshape(b, 2, height_bins, x, y).permute(0,1,3,4,2).contiguous()


class FrozenSGDriveDecoder(FrozenDecoder):
    def __init__(self, decoder, contract):
        super().__init__(decoder)
        self.contract = contract

    def forward(self, hidden):
        if tuple(hidden.shape[1:]) != (self.contract.num_queries,self.contract.hidden_size):
            raise ValueError("SGDrive hidden shape mismatch")
        # The upstream implementation pops both entries. Never reuse this list.
        raw = self.decoder(hidden, [[100,100],[50,50]])
        return sgdrive_logits_layout(raw, self.contract.height_bins)

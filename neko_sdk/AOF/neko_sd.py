# neko_spatial_transform_deform, removes rotation.
# I dunno. May be I am trading fitting capability to variance
import torch
from torch import nn
from torch.nn import functional as trnf
from torchvision import ops as tvop


class neko_scaling_deform_conv_3x3(nn.Module):
    def __init__(self, ifc, ofc, s_hardness=2):
        super(neko_scaling_deform_conv_3x3, self).__init__()
        self.offsets = nn.Parameter(torch.tensor([
            [-1, -1], [-1, 0], [-1, -1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]).float(), requires_grad=False)
        self.dconv = tvop.DeformConv2d(ifc, ofc, 3, 1, 1)
        s_range = (s_hardness - 1 / s_hardness) / 2
        h_ = torch.tensor(
            [s_range, s_range]
        ).reshape([1, 2, 1, 1]).repeat([1, 9, 1, 1])
        self.H = nn.Parameter(h_, requires_grad=False)
        self.ste = torch.nn.Sequential(
            torch.nn.Conv2d(ifc, 18, (1, 1)),
            torch.nn.BatchNorm2d(18),
            torch.nn.Tanh(),
        )

    def make_offset(self, input_feat):
        offs = (self.ste(input_feat) * self.H)
        return offs

    def forward(self, input_feat):
        offs = self.make_offset(input_feat)
        return self.dconv(input_feat, offs)


# Low res: scale and orientation is unlikely to change frequently in a image.
# if they ever change, they change in a contigous way.

class neko_scaling_deform_conv_3x3_lr(neko_scaling_deform_conv_3x3):
    def forward(self, input_feat):
        N, C, H, W = input_feat.shape
        H_ = max(H // 4, 1)
        W_ = max(W // 4, 1)
        sinp = trnf.interpolate(input_feat, [H_, W_], mode="bilinear")
        offs = trnf.interpolate(self.make_offset(sinp), [H, W], mode="bilinear")
        return self.dconv(input_feat, offs)


if __name__ == '__main__':
    magic = neko_scaling_deform_conv_3x3_lr(256, 256)
    a = torch.rand([64, 256, 16, 32])
    b = magic(a)
    pass

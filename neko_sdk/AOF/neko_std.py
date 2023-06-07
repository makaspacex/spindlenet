# neko_spatial_transform_deform
import torch
from torch import nn
from torch.nn import functional as trnf
from torchvision import ops as tvop


class NekoSpatialTransformDeformConv3x3(nn.Module):
    def __init__(self, ifc, ofc, t_hardness=3.14 / 6, s_hardness=2):
        super(NekoSpatialTransformDeformConv3x3, self).__init__()
        self.offsets = nn.Parameter(torch.tensor([
            [-1, -1], [-1, 0], [-1, -1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]).float(), requires_grad=False)
        self.dconv = tvop.DeformConv2d(ifc, ofc, 3, 1, 1)
        s_range = (s_hardness - 1 / s_hardness) / 2
        self.H = nn.Parameter(
            torch.tensor(
                [t_hardness, s_range, s_range]
            ).reshape([1, 3, 1, 1]), requires_grad=False)
        self.B = nn.Parameter(
            torch.tensor(
                [0, s_range, s_range]
            ).reshape([1, 3, 1, 1]), requires_grad=False)
        # theta,scalex,scaley
        self.ste = torch.nn.Sequential(
            torch.nn.Conv2d(ifc, 3, (1, 1)),
            torch.nn.BatchNorm2d(3),
            torch.nn.Tanh(),
        )

    def make_offset(self, input_feat):
        N, C, H, W = input_feat.shape
        info = (self.ste(input_feat) * self.H + self.B).permute(0, 2, 3, 1).reshape(-1, 3)
        sintheta = torch.sin(info[:, 0])
        costheta = torch.cos(info[:, 0:1]) * info[:, 1:3]
        # no translation....
        awt = torch.stack([costheta[:, 0], -sintheta, sintheta, costheta[:, 1]], -1).reshape(N * H * W, 2, 2)
        offs = self.offsets.unsqueeze(0).matmul(awt).reshape(N, H, W, 9 * 2).permute(0, 3, 1, 2)
        return offs

    def forward(self, input_feat):
        offs = self.make_offset(input_feat)
        return self.dconv(input_feat, offs)


# Low res: scale and orientation is unlikely to change frequently in a image.
# if they ever change, they change in a contigous way.

class NekoSpatialTransformDeformConv3x3Lr(NekoSpatialTransformDeformConv3x3):
    def forward(self, input_feat):
        N, C, H, W = input_feat.shape
        H_ = max(H // 4, 1)
        W_ = max(W // 4, 1)
        sinp = trnf.interpolate(input_feat, [H_, W_], mode="bilinear")
        offs = trnf.interpolate(self.make_offset(sinp), [H, W], mode="bilinear")
        return self.dconv(input_feat, offs)


if __name__ == '__main__':
    magic = NekoSpatialTransformDeformConv3x3(256, 256)
    a = torch.rand([64, 256, 16, 32])
    b = magic(a)
    pass

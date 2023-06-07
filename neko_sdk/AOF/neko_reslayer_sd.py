import torch.nn as nn

from neko_sdk.AOF.blocks import BasicBlockNoLens
from neko_sdk.AOF.neko_std import NekoSpatialTransformDeformConv3x3, NekoSpatialTransformDeformConv3x3Lr


# fall back to scaling only.
class NekoSdWrapper(nn.Module):
    def __init__(self, ifc, ofc):
        super(NekoSdWrapper, self).__init__()
        self.core = NekoSpatialTransformDeformConv3x3(ifc, ofc)

    def forward(self, x):
        return self.core(x), None


class NekoSdlrWrapper(nn.Module):
    def __init__(self, ifc, ofc):
        super(NekoSdlrWrapper, self).__init__()
        self.core = NekoSpatialTransformDeformConv3x3Lr(ifc, ofc)

    def forward(self, x):
        return self.core(x), None


class NekoReslayerSd(nn.Module):
    def __init__(self, in_planes, planes, blocks=1, stride=1):
        super(NekoReslayerSd, self).__init__()
        self.in_planes = in_planes
        self.downsample = None
        if stride != 1 or in_planes != planes * BasicBlockNoLens.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlockNoLens.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlockNoLens.expansion),
            )

        self.layers = []
        self.layers.append(BasicBlockNoLens(self.in_planes, planes, stride, self.downsample))
        self.add_module("blk" + "init", self.layers[-1])
        in_planes = planes * BasicBlockNoLens.expansion
        self.layers.append(NekoSdWrapper(in_planes, in_planes))
        self.add_module("SDL", self.layers[-1])
        for i in range(1, blocks):
            self.layers.append(BasicBlockNoLens(in_planes, planes))
            self.add_module("blk" + str(i), self.layers[-1])
        self.out_planes = planes

    def forward(self, input):
        fields = []
        feat = input
        for l in self.layers:
            feat, f = l(feat)
            if (f is not None):
                fields.append(f)
        return feat, fields


class NekoReslayerSdlr(nn.Module):
    def __init__(self, in_planes, planes, blocks=1, stride=1):
        super(NekoReslayerSdlr, self).__init__()
        self.in_planes = in_planes
        self.downsample = None
        if stride != 1 or in_planes != planes * BasicBlockNoLens.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlockNoLens.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlockNoLens.expansion),
            )

        self.layers = []
        self.layers.append(BasicBlockNoLens(self.in_planes, planes, stride, self.downsample))
        self.add_module("blk" + "init", self.layers[-1])
        in_planes = planes * BasicBlockNoLens.expansion
        self.layers.append(NekoSdlrWrapper(in_planes, in_planes))
        self.add_module("STDL", self.layers[-1])
        for i in range(1, blocks):
            self.layers.append(BasicBlockNoLens(in_planes, planes))
            self.add_module("blk" + str(i), self.layers[-1])
        self.out_planes = planes

    def forward(self, input):
        fields = []
        feat = input
        for l in self.layers:
            feat, f = l(feat)
            if (f is not None):
                fields.append(f)
        return feat, fields

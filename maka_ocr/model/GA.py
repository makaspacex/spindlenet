import torch.nn
from torch import nn
from torch.nn import functional as trnf


class SpatialAttention(nn.Module):
    def __init__(self, ifc):
        super(SpatialAttention, self).__init__()
        self.core = torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc, ifc, (3, 3), (1, 1), (1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc),
            torch.nn.Conv2d(ifc, 1, (1, 1)),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        x = input[-1]
        return self.core(x)

class SpatialAttentionMk2(SpatialAttention):

    def forward(self, input):
        x = input[-2]
        d = input[-1]
        if (x.shape[-1] != d.shape[-1]):
            x = trnf.interpolate(x, [d.shape[-2], d.shape[-1]])
        return self.core(x)


class SpatialAttentionMk3(SpatialAttention):
    def forward(self, input):
        x = input[0]
        d = input[-1]
        if (x.shape[-1] != d.shape[-1]):
            x = trnf.interpolate(x, [d.shape[-2], d.shape[-1]], mode="area")
        return self.core(x)


class SpatialAttentionMk4(SpatialAttention):

    def forward(self, input):
        x = input[0]
        d = input[-1]
        y = self.core(x)
        if (y.shape[-1] != d.shape[-1]):
            y = trnf.interpolate(y, [d.shape[-2], d.shape[-1]], mode="bilinear")
        return y


class SpatialAttentionMk5(SpatialAttention):
    def __init__(self, ifc):
        super(SpatialAttention, self).__init__()
        self.core = torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc, ifc, (3, 3), (1, 1), (1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc),
            torch.nn.Conv2d(ifc, 1, (1, 1), bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        x = input[0]
        d = input[-1]
        y = self.core(x)
        if (y.shape[-1] != d.shape[-1]):
            y = trnf.interpolate(y, [d.shape[-2], d.shape[-1]], mode="bilinear")
        return y

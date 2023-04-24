import torch
import torch.nn as nn


# standard backbones(Minimum adaption from torchvision)

class FeatureExtractorHi(nn.Module):
    def __init__(self, strides, compress_layer, input_shape, oupch=512):
        super(FeatureExtractorHi, self).__init__()
        self.model = resnet.resnet45hi(strides, compress_layer, oupch=oupch, inpch=input_shape[0])
        self.input_shape = input_shape

    def freezebn(self):
        self.model.freezebn()

    def unfreezebn(self):
        self.model.unfreezebn()

    def forward(self, input):
        features = self.model(input.contiguous())
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


class FeatureExtractorLo(nn.Module):
    def __init__(self, strides, compress_layer, input_shape, oupch=512):
        super(FeatureExtractorLo, self).__init__()
        self.model = resnet.resnet45lo(strides, compress_layer, oupch=oupch, inpch=input_shape[0])
        self.input_shape = input_shape

    def freezebn(self):
        self.model.freezebn()

    def unfreezebn(self):
        self.model.unfreezebn()

    def forward(self, input):
        features = self.model(input.contiguous())
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

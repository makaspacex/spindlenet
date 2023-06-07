from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neko_sdk.encoders.feat_networks.dropblock import DropBlock


# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_avg_pool = avg_pool
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        return x


def Res12(avg_pool=False, drop_rate=0.1, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, drop_rate=drop_rate, avg_pool=avg_pool, **kwargs)
    return model


class ResNet_fbn(ResNet):
    def _freeze_norm_stats(self):
        try:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        except ValueError:
            print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
            return

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self._freeze_norm_stats()
        return self


def Res12fbn(avg_pool=False, drop_rate=0.1, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, drop_rate=drop_rate, avg_pool=avg_pool, **kwargs)
    return model


class NekoRes12Wrapper(nn.Module):
    def __init__(self, emb_size, keep_prob):
        super(NekoRes12Wrapper, self).__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Conv2d(640, 1, 1),
            torch.nn.Sigmoid(),  # pesudo salience
        )
        self.backbone = Res12(False)
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(640, emb_size),
            nn.Dropout(p=1 - keep_prob, inplace=True)
        )

    def forward(self, images):
        feat = self.backbone(images)
        shape = feat.shape
        ffeat = feat.view(shape[0], shape[1], -1)
        # return self.FC(torch.mean(ffeat,dim=-1))
        #
        sal = self.score(feat) * 0.5 + 0.5
        sal = sal.view(shape[0], 1, -1)
        return self.FC(torch.sum(sal * ffeat, dim=-1) / (torch.sum(sal, dim=-1)))


class NekoPretrainedFeatRes12Wrapper(nn.Module):
    def trim_dict(self, dic):
        d = OrderedDict()
        for k in dic["params"]:
            if k[0:3] == "enc":
                d[k.replace("encoder.", "")] = dic["params"][k]
        return d

    def __init__(self, emb_size, keep_prob, path):
        super(NekoPretrainedFeatRes12Wrapper, self).__init__()
        self.backboneptr = [
            Res12(True, 0.1).cuda()
        ]
        d = self.trim_dict(torch.load(path))
        self.backboneptr[0].load_state_dict(d)
        self.backboneptr[0].eval()
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(640, emb_size),
            nn.Dropout(p=1 - keep_prob, inplace=True)
        )

    def forward(self, images):
        with torch.no_grad():
            feature = self.backboneptr[0](images)
        return self.FC(feature)


class NekoPretrainedFeatRes12WrapperPFT(nn.Module):
    def trim_dict(self, dic):
        d = OrderedDict()
        for k in dic["params"]:
            if k[0:3] == "enc":
                d[k.replace("encoder.", "")] = dic["params"][k]
        return d

    def __init__(self, emb_size, keep_prob, path):
        super(NekoPretrainedFeatRes12WrapperPFT, self).__init__()
        self.backboneptr = Res12(True, 0.1)
        d = self.trim_dict(torch.load(path))
        self.backboneptr.load_state_dict(d)
        # self.backboneptr.eval()
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(640, emb_size),
            nn.Dropout(p=1 - keep_prob, inplace=True)
        )

    def forward(self, images):
        with torch.no_grad():
            feature = self.backboneptr(images)
        featureh = self.backboneptr(images[:3])
        # 0.1 x rl
        feature[:3] = feature[:3] * 0.9 + featureh * 0.1
        return self.FC(feature)


class NekoPretrainedFeatRes12WrapperII(nn.Module):
    def trim_dict(self, dic):
        d = OrderedDict()
        for k in dic["params"]:
            if k[0:3] == "enc":
                d[k.replace("encoder.", "")] = dic["params"][k]
        return d

    def __init__(self, emb_size, keep_prob, path):
        super(NekoPretrainedFeatRes12WrapperII, self).__init__()
        self.backboneptr = [
            Res12(True, 0.1).cuda()
        ]
        d = self.trim_dict(torch.load(path))
        self.backboneptr[0].load_state_dict(d)
        self.backboneptr[0].eval()
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(640, 1280),
            torch.nn.BatchNorm1d(1280),
            torch.nn.PReLU(),
            torch.nn.Linear(1280, emb_size),
            torch.nn.BatchNorm1d(emb_size),
            torch.nn.PReLU(),
            nn.Dropout(p=1 - keep_prob, inplace=True)
        )

    def forward(self, images):
        with torch.no_grad():
            feature = self.backboneptr[0](images)
        return self.FC(feature)


class NekoPretrainedftFeatRes12Wrapper(nn.Module):
    def trim_dict(self, dic):
        d = OrderedDict()
        for k in dic["params"]:
            if k[0:3] == "enc":
                d[k.replace("encoder.", "")] = dic["params"][k]
        return d

    def __init__(self, emb_size, keep_prob, path):
        super(NekoPretrainedftFeatRes12Wrapper, self).__init__()
        self.backbone = Res12(True, 0.1)
        d = self.trim_dict(torch.load(path))
        self.backbone.load_state_dict(d)

        self.FC = torch.nn.Sequential(
            torch.nn.Linear(640, emb_size),
            nn.Dropout(p=1 - keep_prob, inplace=False)
        )

    def forward(self, images):
        with torch.no_grad():
            feature = self.backbone(images)
        return self.FC(feature)


class NekoPretrainedFeatRes12Sap(nn.Module):
    def trim_dict(self, dic):
        d = OrderedDict()
        for k in dic["params"]:
            if k[0:3] == "enc":
                d[k.replace("encoder.", "")] = dic["params"][k]
        return d

    def __init__(self, emb_size, keep_prob, path):
        super(NekoPretrainedFeatRes12Sap, self).__init__()
        self.backboneptr = [
            Res12(False, 0.1).cuda()
        ]
        d = self.trim_dict(torch.load(path))
        self.backboneptr[0].load_state_dict(d)
        self.backboneptr[0].eval()
        self.score = torch.nn.Sequential(
            torch.nn.Conv2d(640, 1, 1),
            torch.nn.Sigmoid(),  # pesudo salience
        )
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(640, emb_size),
            nn.Dropout(p=1 - keep_prob, inplace=False)
        )

    def forward(self, images):
        with torch.no_grad():
            feat = self.backboneptr[0](images)
        shape = feat.shape
        sal = self.score(feat) * 0.5 + 0.5
        sal = sal.view(shape[0], 1, -1)
        ffeat = feat.view(shape[0], shape[1], -1)
        return self.FC(torch.sum(sal * ffeat, dim=-1) / (torch.sum(sal, dim=-1)))


class NekoPretrainedFeatRes12GFC(nn.Module):
    def trim_dict(self, dic):
        d = OrderedDict()
        for k in dic["params"]:
            if k[0:3] == "enc":
                d[k.replace("encoder.", "")] = dic["params"][k]
        return d

    def __init__(self, emb_size, keep_prob, path):
        super(NekoPretrainedFeatRes12GFC, self).__init__()
        self.backboneptr = [
            Res12(False, 0.1).cuda()
        ]
        d = self.trim_dict(torch.load(path))
        self.backboneptr[0].load_state_dict(d)
        self.backboneptr[0].eval()
        self.ds = torch.nn.Sequential(
            torch.nn.Conv2d(640, 128, 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128)  # pesudo salience
        )
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(128 * 5 * 5, emb_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(emb_size),  # pesudo salience
            nn.Dropout(p=1 - keep_prob, inplace=True)
        )

    def forward(self, images):
        with torch.no_grad():
            feat = self.backboneptr[0](images)
        feat = self.ds(feat)
        shape = feat.shape
        ffeat = feat.view(shape[0], -1)
        return self.FC(ffeat)


def neko_feat_Res12_pretrained(emb_size, keep_prob, path="/home/lasercat/ssddata/pretrained/feat/mini/Res12-pre.pth"):
    model = NekoPretrainedFeatRes12Wrapper(emb_size, keep_prob, path)
    return model


def neko_feat_Res12_pretrainedPFT(emb_size, keep_prob,
                                  path="/home/lasercat/ssddata/pretrained/feat/mini/Res12-pre.pth"):
    model = NekoPretrainedFeatRes12WrapperPFT(emb_size, keep_prob, path)
    return model


def neko_feat_Res12_pretrainedII(emb_size, keep_prob, path="/home/lasercat/ssddata/pretrained/feat/mini/Res12-pre.pth"):
    model = NekoPretrainedFeatRes12WrapperII(emb_size, keep_prob, path)
    return model


def neko_feat_Res12_pretrained_ft(emb_size, keep_prob,
                                  path="/home/lasercat/ssddata/pretrained/feat/mini/Res12-pre.pth"):
    model = NekoPretrainedftFeatRes12Wrapper(emb_size, keep_prob, path)
    return model


def neko_feat_Res12_pretrained_sap(emb_size, keep_prob,
                                   path="/home/lasercat/ssddata/pretrained/feat/mini/Res12-pre.pth"):
    model = NekoPretrainedFeatRes12Sap(emb_size, keep_prob, path)
    return model


def neko_feat_Res12_pretrained_GFC(emb_size, keep_prob,
                                   path="/home/lasercat/ssddata/pretrained/feat/mini/Res12-pre.pth"):
    model = NekoPretrainedFeatRes12GFC(emb_size, keep_prob, path)
    return model


def neko_Res12(emb_size, keep_prob=0.5, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = NekoRes12Wrapper(emb_size, keep_prob)

    return model

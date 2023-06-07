# from neko_sdk.encoders.feat_networks.ires import conv_iResNet
import torch
from torch import nn

from neko_sdk.encoders.feat_networks.ires import ConvIResNet
from neko_sdk.encoders.ocr_networks.neko_pyt_resnet_np import resnet18np
# from  torchvision.models import resnet18,resnet34
from neko_sdk.encoders.tv_res_nip import resnet18, resnet34


class NekoVisualOnlyInterprinter(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(NekoVisualOnlyInterprinter, self).__init__()
        if core is None:
            self.core = resnet18(num_classes=feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"]
        vp = self.core(view_dict)

        # print(nvp.norm(dim=1))
        return vp


class MagicCore(nn.Module):
    def __init__(self, feature_cnt):
        super(MagicCore, self).__init__()
        self.c = ConvIResNet([3, 32, 32], [2, 2, 2, 2], [1, 2, 2, 2], [32, 32, 32, 32],
                             init_ds=2, density_estimation=False, actnorm=True)
        self.f = torch.nn.Linear(768, feature_cnt, False)
        self.d = torch.nn.Dropout(0.1)

    def forward(self, x):
        c = self.c(x)
        c = c.mean(dim=(2, 3))
        p = self.f(c)
        return self.d(p)


# class neko_visual_only_interprinter_inv(nn.Module):
#     def __init__(self,feature_cnt,core=None):
#         super(neko_visual_only_interprinter_inv,self).__init__()
#         if core is None:
#             self.core=magic_core(feature_cnt)
#         else:
#             self.core=core
#     def forward(self,view_dict) :
#         # vis_proto=view_dict["visual"]
#         vp=self.core(view_dict)
#
#         # print(nvp.norm(dim=1))
#         return vp
class NekoVisualOnlyInterprinterHD(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(NekoVisualOnlyInterprinterHD, self).__init__()
        if core is None:
            self.core = resnet18np(outch=feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"]
        vp = self.core(view_dict).permute(0, 2, 3, 1).reshape(view_dict.shape[0], -1)
        # print(nvp.norm(dim=1))
        return vp


class NekoVisualOnlyInterprinterR34(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(NekoVisualOnlyInterprinterR34, self).__init__()
        if core is None:
            self.core = resnet34(num_classes=feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"]
        vp = self.core(view_dict)
        # print(nvp.norm(dim=1))
        return vp


class NekoStructuralVisualOnlyInterprinter(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(NekoStructuralVisualOnlyInterprinter, self).__init__()
        if core is None:
            self.core = resnet18np(outch=feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"]
        vp = self.core(view_dict)
        return vp.view(vp.shape[0], -1)
        # print(nvp.norm(dim=1))

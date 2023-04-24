import torch
from torch import nn


# PROTOENGINE = neko_visual_only_interprinter


class neko_prototyper_sp(nn.Module):
    def __init__(self, output_channel, spks, has_sem=False):
        super(neko_prototyper_sp, self).__init__()
        self.output_channel = output_channel
        self.sp_cnt = len(spks)
        self.EOS = 0
        self.sp_protos_vis = torch.nn.Parameter(torch.rand([
            self.sp_cnt, self.output_channel]).float() * 2 - 1)
        self.register_parameter("sp_protos_vis", self.sp_protos_vis)
        if (has_sem):
            self.sp_protos_sem = torch.nn.Parameter(torch.rand([
                self.sp_cnt, self.output_channel]).float() * 2 - 1)
            self.register_parameter("sp_protos_sem", self.sp_protos_sem)
        else:
            self.sp_protos_sem = None

    def forward(self):
        return self.sp_protos_vis, self.sp_protos_sem

# # We do NOT reduce visual feature in this module. In fact,
# # a cam shall follow the backbone to reduce it thru w and h.
# # the rotation has to be done somewhere else.
# class neko_prototyperpost(nn.Module):
#     def __init__(self, output_channel, spks, dropout=None, capacity=512):
#         super(neko_prototyperpost, self).__init__()
#         self.output_channel = output_channel
#         self.sp_cnt = len(spks)
#         self.proto_engine = self.PROTOENGINE(self.output_channel)
#         self.dev_ind = torch.nn.Parameter(torch.rand([1]))
#         self.EOS = 0
#         self.sp_protos = torch.nn.Parameter(torch.rand([
#             self.sp_cnt, self.output_channel]).float() * 2 - 1)
#         self.register_parameter("sp_proto", self.sp_protos)
#         if (dropout is not None):
#             self.drop = torch.nn.Dropout(p=0.3)
#         else:
#             self.drop = None
#         print("DEBUG-SDFGASDFGSDGASFGSD", dropout)
#         # split if too many
#         self.capacity = capacity
#         self.freeze_bn_affine = False
#
#     def freezebn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()
#                 if self.freeze_bn_affine:
#                     m.weight.requires_grad = False
#                     m.bias.requires_grad = False
#
#     def unfreezebn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.train()
#                 if self.freeze_bn_affine:
#                     m.weight.requires_grad = True
#                     m.bias.requires_grad = True
#
#     def forward(self, vis_proto_srcs,sem_proto_srcs):
#         allproto = trnf.normalize(torch.cat(vis_proto_srcs), dim=1, eps=0.0009)
#         if (self.drop):
#             allproto = self.drop(allproto)
#         pass
#         return allproto.contiguous()

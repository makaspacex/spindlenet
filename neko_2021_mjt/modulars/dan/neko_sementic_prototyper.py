import torch
from torch.nn import functional as trnf


# note that this version is not dict driven---
# Since torch.module does not provide non-numeric parameter support,
# we offload the labelstr-id mapping to sampler
# This class allows you to get a subset via the sid given by the label sampler, if applicable.
# However, FSL datasets may sample label in dataloaders, which means the sid is needed to be provided by the loader, if necessary.

class neko_sampled_sementic_branch(torch.nn.Module):
    def __init__(self, feat_ch, capacity, spks):
        super(neko_sampled_sementic_branch, self).__init__()
        self.weights = torch.nn.Parameter(torch.rand([capacity, feat_ch]) * 2 - 1)

    def sample(self, sids):
        ret = []
        for i in sids:
            ret.append(self.weights[i])
        return trnf.normalize(torch.stack(ret), dim=-1)

    def forward(self, sids=None):
        if (sids is None):
            return trnf.normalize(self.weights, dim=-1)
        else:
            return self.sample(sids)

#
# class neko_global_sementic_branch(neko_sampled_sementic_branch):
#     def forward(self,):
#     gids=[]
#         for i in plabel:
#             gids.append(gtdict[tdict[i]])
#         return self.sample(gids)

import torch

from neko_sdk.ocr_modules.neko_score_merging import scatter_cvt


class neko_openset_linear_classifier(torch.nn.Module):
    #    n=nB or n=nB*nT   nxnC       kxnC
    def __init__(self):
        super(neko_openset_linear_classifier, self).__init__()
        self.UNK_SCR = torch.nn.Parameter(torch.zeros(1).float())

    def get_unk_scr(self, flat_emb):
        return self.UNK_SCR.expand(flat_emb.shape[0], 1)

    def get_scr(self, flat_emb, protos):
        return flat_emb.matmul(protos.T)

    def forward(self, flat_emb, protos, plabel):
        out_res_ = torch.cat([self.get_scr(flat_emb, protos), self.get_unk_scr(flat_emb)], -1)
        if (plabel is not None):
            scores = scatter_cvt(out_res_, plabel)
        else:
            scores = out_res_
        return scores


class neko_openset_linear_classifierK(neko_openset_linear_classifier):
    def get_unk_scr(self, flat_emb):
        return self.UNK_SCR * flat_emb.norm(dim=-1, keepdim=True)

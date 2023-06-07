import torch
import torch_scatter
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.ocr_modules.trainable_losses.cosloss import NekoCosLoss2

try:
    import pylcs
except:
    pylcs = None
    print("no pylcs!, some loss (trident net) won't work!")


class OsdanLoss(nn.Module):
    def __init__(self, cfgs):
        super(OsdanLoss, self).__init__()
        self.setuploss(cfgs)

    def label_weight(self, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float())
        weight = 1. / weight
        weight[-1] /= 200
        return weight

    def setuploss(self, cfgs):
        # self.aceloss=
        self.cosloss = NekoCosLoss2().cuda()
        self.wcls = cfgs["wcls"]
        self.wsim = cfgs["wsim"]
        self.wemb = cfgs["wemb"]
        self.wmar = cfgs["wmar"]

    def forward(self, proto, outcls, outcos, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float()
        w[-1] = 0.1
        # change introduced with va5. Masked timestamp does not contribute to loss.
        # Though we cannot say it's unknown(if the image contains one single character) --- perhaps we can?
        clsloss = trnf.cross_entropy(outcls, label_flatten, w, ignore_index=-1)
        if (self.wmar > 0):
            margin_loss = self.url.forward(outcls, label_flatten, 0.5)
        else:
            margin_loss = 0

        if (outcos is not None and self.wsim > 0):
            cos_loss = self.cosloss(outcos, label_flatten)
            # ace_loss=self.aceloss(outcls,label_flatten)
            loss = cos_loss * self.wsim + clsloss * self.wcls + margin_loss * self.wmar + self.wemb * proto_loss
            terms = {
                "total": loss.detach().item(),
                "margin": margin_loss.detach().item(),
                "main": clsloss.detach().item(),
                "sim": cos_loss.detach().item(),
                "emb": proto_loss.detach().item(),
            }
        else:
            loss = clsloss * self.wcls + margin_loss * self.wmar + self.wemb * proto_loss
            terms = {
                "total": loss.detach().item(),
                # "margin": margin_loss.detach().item(),
                "main": clsloss.detach().item(),
                "emb": proto_loss.detach().item(),
            }
        return loss, terms


class OsdanLossClsctx(nn.Module):
    def __init__(self, cfgs):
        super(OsdanLossClsctx, self).__init__()
        self.setuploss(cfgs)

    def setuploss(self, cfgs):
        self.criterion_CE = nn.CrossEntropyLoss()
        # self.aceloss=
        self.wcls = cfgs["wcls"]
        self.wctx = cfgs["wctx"]
        self.wshr = cfgs["wshr"]
        self.wpemb = cfgs["wpemb"]
        self.wgemb = cfgs["wgemb"]
        self.reduction = cfgs["reduction"]

    def forward(self, proto, outcls, ctxproto, ctxcls, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float()
        w[-1] = 0.1
        clsloss = trnf.cross_entropy(outcls, label_flatten, w, ignore_index=-1)
        loss = clsloss * self.wcls + self.wemb * proto_loss
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms


class OsdanLossClsemb(nn.Module):
    def __init__(self, cfgs):
        super(OsdanLossClsemb, self).__init__()
        self.setuploss(cfgs)

    def label_weight(self, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float())
        weight = 1. / weight
        weight[-1] /= 200
        return weight

    def setuploss(self, cfgs):
        self.criterion_CE = nn.CrossEntropyLoss()
        # self.aceloss=
        self.wcls = cfgs["wcls"]
        self.wemb = cfgs["wemb"]
        self.reduction = cfgs["reduction"]

    def forward(self, proto, outcls, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float()
        w[-1] = 0.1
        clsloss = trnf.cross_entropy(outcls, label_flatten, w, ignore_index=-1)
        loss = clsloss * self.wcls + self.wemb * proto_loss
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms


class FsldanlossClsembohem(nn.Module):
    def __init__(self, cfgs):
        super(FsldanlossClsembohem, self).__init__()
        self.setuploss(cfgs)

    def setuploss(self, cfgs):
        self.criterion_CE = nn.CrossEntropyLoss()
        # self.aceloss=
        self.wcls = cfgs["wcls"]
        self.wemb = cfgs["wemb"]
        self.dirty_frac = cfgs["dirty_frac"]
        self.too_simple_frac = cfgs["too_simple_frac"]

    def forward(self, proto, outcls, label_flatten):
        if (self.wemb > 0):
            proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        else:
            proto_loss = torch.tensor(0.).float()
        clsloss = trnf.cross_entropy(outcls, label_flatten, reduction="none")
        with torch.no_grad():
            w = torch.ones_like(clsloss, device=proto.device).float()
            # 20% dirty data. The model drops 20% samples that show less agreement to history experience
            # These predictions are expected to be handled by the ctx module following.
            tpk = int(clsloss.shape[0] * self.too_simple_frac)
            if (tpk > 0):
                w[torch.topk(clsloss, tpk, 0, largest=False)[1]] = 0
                w[clsloss > 0.5] = 1
            w[torch.topk(clsloss, int(clsloss.shape[0] * self.dirty_frac), 0, largest=True)[1]] = 0
            # w[label_flatten==proto.shape[0]]=1
        clsloss = (w * clsloss).sum() / (w.sum() + 0.00001)
        loss = clsloss * self.wcls + self.wemb * proto_loss
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms


class OsdanlossTrident(nn.Module):
    def __init__(self, cfgs):
        super(OsdanlossTrident, self).__init__()
        self.setuploss(cfgs)

    def label_weight(self, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float())
        weight = 1. / weight
        weight[-1] /= 200
        return weight

    def setuploss(self, cfgs):
        # self.aceloss=
        self.wcls = cfgs["wcls"]
        self.wemb = cfgs["wemb"]
        self.wrew = cfgs["wrew"]
        self.ppr = cfgs["ppr"]

    def get_scatter_region(self, labels, length):
        target = torch.zeros(length)
        beg = 0
        id = 0
        for l in labels:
            le = len(l) + 1
            target[beg:beg + le] = id
            id += 1
            beg += le
        return target

    def compute_rewards(self, choutputs, labels):
        rew = torch.zeros(len(labels), len(choutputs))
        for i in range(len(labels)):
            for j in range(len(choutputs)):
                rew[i][j] = 1 - (pylcs.edit_distance(choutputs[j][i], labels[i])) / (len(labels[i]) + 0.0001)
        # minus baseline trick, seems I forgot to detach something(www)
        # TODO: check whether we need to detach the mean. 
        rew[:, 1:] -= 0.02  # prefer the square option(which does not require encoding the picture again),
        # however if the situation requires invoke the necessary branch
        rrew = rew - rew.mean(dim=1, keepdim=True)

        return rrew

    def forward(self, proto, outclss, outcoss, choutputs, branch_logit, branch_prediction, labels, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        w = torch.ones_like(torch.ones(outclss[0].shape[-1])).to(proto.device).float()
        w[-1] = 0.1
        sr = self.get_scatter_region(labels, outclss[0].shape[0])
        cls_losses = []
        rew = self.compute_rewards(choutputs, labels)
        for i in range(len(outclss)):
            clsloss = trnf.cross_entropy(outclss[i], label_flatten, w, reduction="none")
            scl = torch_scatter.scatter_mean(clsloss, sr.long().cuda())
            cls_losses.append(scl)
        # ace_loss=self.aceloss(outcls,label_flatten)
        # Vodoo to keep each branch alive. 1.9=0.3+1
        tw = (1 + self.ppr * len(outclss))
        weight = (branch_prediction.detach() + self.ppr) / tw

        clslossw = (torch.stack(cls_losses, 1) * weight).sum(1).mean()

        arew = (rew.cuda() * branch_prediction).sum(1).mean()
        loss = clslossw * self.wcls + self.wemb * proto_loss - self.wrew * arew
        terms = {
            "total": loss.detach().item(),
            "main": clslossw.detach().item(),
            "emb": proto_loss.detach().item(),
            "Exp. 1-ned": arew.detach().item(),
        }
        return loss, terms

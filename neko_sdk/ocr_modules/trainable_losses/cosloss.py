import torch
from torch import nn
from torch.nn import functional as trnf


class NekoCosLoss(nn.Module):
    def __init__(self):
        super(NekoCosLoss, self).__init__()
        pass

    def forward(self, pred, gt):
        oh = torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)[:, :-1]
        noh = 1 - oh
        pred_ = pred[:, :-1]
        pwl = torch.nn.functional.smooth_l1_loss(pred_, oh, reduction="none")
        nl = torch.sum(noh * pwl) / (torch.sum(noh) + 0.009)
        pl = torch.sum(oh * pwl) / (torch.sum(oh) + 0.009)
        return (nl + pl) / 2


class NekoCosLossx(nn.Module):
    def __init__(self):
        super(NekoCosLossx, self).__init__()
        pass

    def forward(self, pred, gt):
        oh = torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)
        noh = 1 - oh
        pwl = trnf.l1_loss(pred, oh, reduction="none")
        nl = (noh * pwl).reshape(-1).topk(int(torch.sum(oh).item() + 1))[0].mean()
        pl = torch.sum(oh * pwl) / (torch.sum(oh) + 0.009)
        return (nl + pl) / 2


class NekoCosLoss2ar(nn.Module):
    def __init__(self):
        super(NekoCosLoss2ar, self).__init__()
        pass

    def forward(self, pred, gt):
        oh = torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)
        noh = 1 - oh
        pwl = trnf.l1_loss(pred, oh, reduction="none")
        nwl = trnf.l1_loss(trnf.relu(pred), oh, reduction="none")
        nohc = noh + 0
        ohc = oh + 0
        nohc[:, -1] *= 0.1
        ohc[:, -1] *= 0.1
        nl = torch.sum(nohc * nwl) / (torch.sum(nohc) + 0.009)
        pl = torch.sum(ohc * pwl) / (torch.sum(ohc) + 0.009)
        return (nl + pl) / 2


class NekoCosLoss2a(nn.Module):
    def __init__(self):
        super(NekoCosLoss2a, self).__init__()
        pass

    def forward(self, pred, gt):
        pred = trnf.relu(pred)
        oh = torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)
        noh = 1 - oh
        pwl = trnf.l1_loss(pred, oh, reduction="none")
        nohc = noh + 0
        ohc = oh + 0
        nohc[:, -1] *= 0.1
        ohc[:, -1] *= 0.1
        nl = torch.sum(nohc * pwl) / (torch.sum(nohc) + 0.009)
        pl = torch.sum(ohc * pwl) / (torch.sum(ohc) + 0.009)
        return (nl + pl) / 2


class NekoCosLoss2f(nn.Module):
    def __init__(self):
        super(NekoCosLoss2f, self).__init__()
        pass

    def forward(self, pred, gt):
        w = torch.ones_like(torch.ones(pred.shape[-1])).to(pred.device).float()
        w[-1] = 0.1
        # change introduced with va5. Masked timestamp does not contribute to loss.
        # Though we cannot say it's unknown(if the image contains one single character) --- perhaps we can?
        clsloss = trnf.cross_entropy(pred, gt, w, ignore_index=-1)
        return clsloss


class NekoCosLoss2(nn.Module):
    def __init__(self):
        super(NekoCosLoss2, self).__init__()
        pass

    def forward(self, pred, gt):
        oh = torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)[:, :-1]
        noh = 1 - oh
        pred_ = pred[:, :-1]
        corr = oh * pred_

        ### Only classes too close to each other should be pushed.
        ### That said if they are far we don't need to align them
        ### Again 0.14 =cos(spatial angluar on 50k evenly distributed prototype)
        wrong = torch.nn.functional.relu(noh * pred_ - 0.14)
        nl = torch.sum(wrong) / (torch.sum(noh) + 0.009)
        pl = 1 - torch.sum(corr) / (torch.sum(oh) + 0.009)
        return (nl + pl) / 2

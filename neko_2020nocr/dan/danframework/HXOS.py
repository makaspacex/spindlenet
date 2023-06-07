# ------------------------
import os

from neko_2020nocr.dan.common.common import Zero_Grad, Train_or_Eval, flatten_label
from neko_2020nocr.dan.common.common_xos import load_network
from neko_2020nocr.dan.danframework.neko_abstract_dan import NekoAbstractDan
from neko_2020nocr.dan.utils import *
from neko_2020nocr.dan.visdan import Visdan
from neko_sdk.ocr_modules.neko_confusion_matrix import NekoConfusionMatrix
# from torch_scatter import scatter_mean
from neko_sdk.ocr_modules.trainable_losses.cosloss import NekoCosLoss
from neko_sdk.ocr_modules.trainable_losses.neko_url import NekoUnknownRankingLoss
from torch import nn


class NekoCosLoss3(nn.Module):
    def __init__(self):
        super(NekoCosLoss3, self).__init__()
        pass

    def forward(self, pred, gt, weight=None):
        oh = torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)[:, 1:-1]
        noh = 1 - oh
        # oh*=weight[1:-1].unsqueeze(0)
        # noh *= weight[1:-1].unsqueeze(0)
        mask = oh.max(dim=-1, keepdim=True)[0]
        oh *= mask
        noh *= mask
        pred_ = pred[:, 1:-1]
        corr = oh * pred_
        ### Only classes too close to each other should be pushed.
        ### That said if they are far we don't need to align them
        ### Again 0.14 =cos(spatial angluar on 50k evenly distributed prototype)
        wrong = torch.nn.functional.relu(noh * pred_ - 0.14)
        hwrong = torch.max(torch.nn.functional.relu(noh * pred_ - 0.14), dim=1)[0]

        nl = torch.sum(wrong) / (torch.sum(noh) + 0.009) + hwrong.mean()
        pl = 1 - torch.sum(corr) / (torch.sum(oh) + 0.009)
        return (nl + pl) / 2


# HSOS, HDOS, HDOSCS
# the main feature is that the DPE returns embeddings for characters.
class HXOS(NekoAbstractDan):
    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()

    def load_network(self):
        return load_network(self.cfgs)

    def get_ar_cntr(self, key, case_sensitive):
        return NekoOsAttentionArCounter(key, case_sensitive)

    def get_loss_cntr(self, show_interval):
        return LossCounter(show_interval)

    def test(self, test_loader, model, tools, miter=1000, debug=False, dbgpath=None):
        Train_or_Eval(model, 'Eval')
        proto, semb, plabel, tdict = model[3].dump_all()
        i = 0
        visualizer = None
        if dbgpath is not None:
            visualizer = Visdan(dbgpath)
        cfm = NekoConfusionMatrix()

        for sample_batched in test_loader:
            if i > miter:
                break
            i += 1
            data = sample_batched['image']
            label = sample_batched['label']
            target = model[3].encode(proto, plabel, tdict, label)

            data = data.cuda()
            target = target
            label_flatten, length = tools[1](target)
            target, label_flatten = target.cuda(), label_flatten.cuda()

            features = model[0](data)
            A = model[1](features)
            # A0=A.detach().clone()
            output, out_length, A = model[2](features[-1], proto, semb, plabel, A, None, length, True)
            # A=A.max(dim=2)[0]
            choutput, prdt_prob = model[3].decode(output, out_length, proto, plabel, tdict)
            tools[0].add_iter(choutput, out_length, label, debug)

            for i in range(len(choutput)):
                cfm.addpairquickandugly(choutput[i], label[i])

            if (visualizer is not None):
                visualizer.addbatch(data, A, label, choutput)
        if (dbgpath):
            try:
                cfm.save_matrix(os.path.join())
            except:
                pass
        tools[0].show()
        Train_or_Eval(model, 'Train')

    def runtest(self, miter=1000, debug=False, dpgpath=None):
        self.test((self.test_loader), self.model, [self.test_acc_counter,
                                                   flatten_label,
                                                   ], miter=miter, debug=debug, dbgpath=dpgpath)
        self.test_acc_counter.clear()

    def mk_proto(self, label):
        return None, None, None, None

    def fpbp(self, data, label, cased=None):
        proto, semb, plabel, tdict = self.mk_proto(label)
        target = self.model[3].encode(proto, plabel, tdict, label)

        Train_or_Eval(self.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = self.model[0](data)
        A = self.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        output, _ = self.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = self.model[3].decode(output, length, proto, plabel, tdict)
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        self.train_acc_counter.add_iter(choutput, length, tarswunk)
        loss = self.criterion_CE(output, label_flatten)
        self.loss_counter.add_iter(loss)
        # update network
        Zero_Grad(self.model)
        loss.backward()


class HSOS(HXOS):
    def mk_proto(self, label):
        return self.model[3].dump_all()


class HDOS(HXOS):
    def mk_proto(self, label):
        return self.model[3].sample_tr(label)


class HXOSC(HXOS):
    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()
        self.cosloss = NekoCosLoss().cuda()

    def fpbp(self, data, label, cased=None):
        proto, semb, plabel, tdict = self.mk_proto(label)
        target = self.model[3].encode(proto, plabel, tdict, label)

        Train_or_Eval(self.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = self.model[0](data)
        A = self.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = self.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = self.model[3].decode(outcls, length, proto, plabel, tdict)
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        self.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = self.criterion_CE(outcls, label_flatten)
        cos_loss = self.cosloss(outcos, label_flatten)
        loss = cos_loss + clsloss
        self.loss_counter.add_iter(loss)
        # update network
        Zero_Grad(self.model)
        loss.backward()


class HDOSC(HXOSC):
    def mk_proto(self, label):
        return self.model[3].sample_tr(label)


class HXOSCR(HXOSC):
    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()
        self.url = NekoUnknownRankingLoss()
        self.cosloss = NekoCosLoss().cuda()
        self.wcls = 1
        self.wsim = 1
        self.wmar = 0

    def fpbp(self, data, label, cased=None):
        proto, semb, plabel, tdict = self.mk_proto(label)
        target = self.model[3].encode(proto, plabel, tdict, label)

        Train_or_Eval(self.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = self.model[0](data)
        A = self.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = self.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = self.model[3].decode(outcls, length, proto, plabel, tdict)

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        self.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = self.criterion_CE(outcls, label_flatten)
        cos_loss = self.cosloss(outcos, label_flatten)
        margin_loss = self.url.forward(outcls, label_flatten, 0.5)
        loss = cos_loss * self.wsim + clsloss * self.wcls + margin_loss * self.wmar
        terms = {
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim": cos_loss.detach().item(),
        }
        self.loss_counter.add_iter(loss, terms)
        # update network
        Zero_Grad(self.model)
        loss.backward()


class HDOSCR(HXOSCR):
    def mk_proto(self, label):
        return self.model[3].sample_tr(label)


class HXOSCRR(HXOSC):
    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()
        self.url = NekoUnknownRankingLoss()
        self.cosloss = NekoCosLoss().cuda()
        self.wcls = 1
        self.wsim = 1
        self.wmar = 0.3

    def fpbp(self, data, label, cased=None):
        proto, semb, plabel, tdict = self.mk_proto(label)
        target = self.model[3].encode(proto, plabel, tdict, label)

        Train_or_Eval(self.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = self.model[0](data)
        A = self.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = self.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = self.model[3].decode(outcls, length, proto, plabel, tdict)

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        self.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = self.criterion_CE(outcls, label_flatten)
        cos_loss = self.cosloss(outcos, label_flatten)
        margin_loss = self.url.forward(outcls, label_flatten, 0.5)
        loss = cos_loss * self.wsim + clsloss * self.wcls + margin_loss * self.wmar
        terms = {
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim": cos_loss.detach().item(),
        }
        self.loss_counter.add_iter(loss, terms)
        # update network
        Zero_Grad(self.model)
        loss.backward()


class HDOSCRR(HXOSCRR):
    def mk_proto(self, label):
        return self.model[3].sample_tr(label)


class HXOSCO(HXOSC):
    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()
        self.url = NekoUnknownRankingLoss()
        self.cosloss = NekoCosLoss().cuda()
        self.wcls = 0
        self.wsim = 1
        self.wmar = 0

    def fpbp(self, data, label, cased=None):
        proto, semb, plabel, tdict = self.mk_proto(label)
        target = self.model[3].encode(proto, plabel, tdict, label)

        Train_or_Eval(self.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = self.model[0](data)
        A = self.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = self.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = self.model[3].decode(outcls, length, proto, plabel, tdict)

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        self.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = self.criterion_CE(outcls, label_flatten)
        cos_loss = self.cosloss(outcos, label_flatten)
        margin_loss = self.url.forward(outcls, label_flatten, 0.5)
        loss = cos_loss * self.wsim + clsloss * self.wcls + margin_loss * self.wmar
        terms = {
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim": cos_loss.detach().item(),
        }
        self.loss_counter.add_iter(loss, terms)
        # update network
        Zero_Grad(self.model)
        loss.backward()


class HDOSCO(HXOSCRR):
    def mk_proto(self, label):
        return self.model[3].sample_tr(label)


class HXOSCB(HXOSC):
    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()
        self.url = NekoUnknownRankingLoss()
        self.cosloss = NekoCosLoss().cuda()
        self.wcls = 1
        self.wsim = 0
        self.wmar = 0

    def fpbp(self, data, label, cased=None):
        proto, semb, plabel, tdict = self.mk_proto(label)
        target = self.model[3].encode(proto, plabel, tdict, label)

        Train_or_Eval(self.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = self.model[0](data)
        A = self.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = self.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = self.model[3].decode(outcls, length, proto, plabel, tdict)

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        self.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = self.criterion_CE(outcls, label_flatten)
        cos_loss = self.cosloss(outcos, label_flatten)
        margin_loss = self.url.forward(outcls, label_flatten, 0.5)
        loss = cos_loss * self.wsim + clsloss * self.wcls + margin_loss * self.wmar
        terms = {
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim": cos_loss.detach().item(),
        }
        self.loss_counter.add_iter(loss, terms)
        # update network
        Zero_Grad(self.model)
        loss.backward()


class HDOSB(HXOSCB):
    def mk_proto(self, label):
        return self.model[3].sample_tr(label)

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.ocr_modules.neko_interprinter import NekoVisualOnlyInterprinter, NekoVisualOnlyInterprinterR34
from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_abstractract_sampler import NekoPrototypeSamplerStatic


class NekoPrototyper(nn.Module):
    PROTOENGINE = NekoVisualOnlyInterprinter

    def __init__(self, output_channel, spks, dropout=None, capacity=512):
        super(NekoPrototyper, self).__init__()
        self.output_channel = output_channel
        self.sp_cnt = len(spks)
        self.proto_engine = self.PROTOENGINE(self.output_channel)
        self.dev_ind = torch.nn.Parameter(torch.rand([1]))
        self.EOS = 0
        self.sp_protos = torch.nn.Parameter(torch.rand([
            self.sp_cnt, self.output_channel]).float() * 2 - 1)
        self.register_parameter("sp_proto", self.sp_protos)
        if (dropout is not None):
            self.drop = torch.nn.Dropout(p=0.3)
        else:
            self.drop = None
        print("DEBUG-SDFGASDFGSDGASFGSD", dropout)
        # split if too many
        self.capacity = capacity
        self.freeze_bn_affine = False

    def freezebn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if self.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def unfreezebn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if self.freeze_bn_affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True

    def forward(self, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= self.capacity):
            # pimage=torch.cat(normprotos).to(self.dev_ind.device)
            pimage = torch.cat(normprotos).contiguous().to(self.dev_ind.device)

            if (rot > 0):
                pimage = torch.rot90(pimage, rot, [2, 3])

            if (pimage.shape[1] == 1):
                pimage = pimage.repeat([1, 3, 1, 1])
            if (use_sp):
                proto = [self.sp_protos, self.proto_engine(pimage)]
            else:
                proto = [self.proto_engine(pimage)]
        else:
            if (use_sp):
                proto = [self.sp_protos]
            else:
                proto = []
            for s in range(0, len(normprotos), self.capacity):
                pimage = torch.cat(normprotos[s:s + self.capacity]).contiguous().to(self.dev_ind.device)
                if (rot > 0):
                    pimage = torch.rot90(pimage, rot, [2, 3])
                if (pimage.shape[1] == 1):
                    pimage = pimage.repeat([1, 3, 1, 1])
                    proto.append(self.proto_engine(pimage))

        allproto = trnf.normalize(torch.cat(proto), dim=1, eps=0.0009)
        if (self.drop):
            allproto = self.drop(allproto)
        pass
        return allproto.contiguous()


class NekoPrototyperR34(NekoPrototyper):
    PROTOENGINE = NekoVisualOnlyInterprinterR34


class NekoPrototypeSamplerBasic(NekoPrototypeSamplerStatic):
    def train(self, training=True):
        pass

    def eval(self):
        pass

    def cuda(self):
        pass

    # defines sampler
    def setup_sampler(self, sampler_args):
        if sampler_args is None:
            max_match_size = 512
            val_frac = 0.8
            neg_servant = True
        else:
            max_match_size = sampler_args["max_batch_size"]
            val_frac = sampler_args["val_frac"]
            neg_servant = sampler_args["neg_servant"]
        self.max_batch_size = max_match_size
        self.val_frac = val_frac
        self.neg_servant = neg_servant

    def debug(self, normpids, labels):
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in normpids]
        protos = ((torch.cat(normprotos, dim=-1).squeeze(0).NekoSqueeze(0) + 1) * 127.5).detach().cpu().numpy().astype(
            np.uint8)
        import cv2
        cv2.imshow(labels, protos[:, :32 * 32])
        cv2.waitKey(0)

    def grab_cluster(self, ch):
        chid = self.label_dict[ch]
        ret = {chid}
        if self.masters_share:
            ret.add(self.masters[chid])
            ret = ret.union(self.servants[self.masters[chid]])
        return ret

    def dump_all_impl(self, use_sp=True):
        if (use_sp):
            trsps = [self.EOS]
        else:
            trsps = []
        trchs = list(set([self.label_dict[i] for i in self.shaped_characters]))
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return normprotos, plabels, tdicts

    def get_sampled_ids(self, plain_chars_in_data):
        cntval = int(len(plain_chars_in_data) * self.val_frac)
        cntval = min(self.max_batch_size - self.sp_cnt, cntval)
        trchs = set()
        related_chars_in_data = set()
        random.shuffle(plain_chars_in_data)
        # make sure no missing centers--
        # or it may enforce "A" to look like "a" encoded by proto CNN
        remaining = cntval
        for ch in plain_chars_in_data:
            if (ch not in self.label_dict):
                continue
            new = self.grab_cluster(ch)
            ns = trchs.union(new)
            related_chars_in_data = related_chars_in_data.union(new)
            delta = len(ns) - len(trchs)
            if (delta <= remaining):
                trchs = ns
                remaining -= delta
        remaining = self.max_batch_size - self.sp_cnt - len(trchs)
        plain_charid_not_in_data = list(self.shaped_ids - related_chars_in_data)
        random.shuffle(plain_charid_not_in_data)
        for chid in plain_charid_not_in_data:
            if chid not in trchs:
                if (remaining == 0):
                    break
                if (self.neg_servant == False and self.masters[chid] != chid):
                    continue
                remaining -= 1
                trchs.add(chid)

        trsps = set([self.label_dict[i] for i in self.sp_tokens])
        return trsps, trchs

    def sample_charset_by_text(self, text_batch, use_sp=True):
        plain_chars_in_data = self.get_occured(text_batch)
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        if (use_sp is not False):
            trsps = list(trsps)
        else:
            trsps = []
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        # self.debug(trchs,"meow")
        return normprotos, plabels, tdicts

    def sample_charset_by_textg(self, text_batch, use_sp=True):
        plain_chars_in_data = self.get_occured(text_batch)
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        if (use_sp is not False):
            trsps = list(trsps)
        else:
            trsps = []
        plabels, tdicts, gtdicts = self.get_plabel_and_dictg(trsps, trchs)
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        # self.debug(trchs,"meow")
        return normprotos, plabels, tdicts, gtdicts

    def sample_charset_by_text2(self, text_batch):
        plain_chars_in_data = self.get_occured(text_batch)
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        trsps = list(trsps)
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        # self.debug(trchs,"meow")
        return normprotos, plabels, tdicts

    def sample_charset_by_text_both(self, text_batch):
        b = ""
        for _ in text_batch: b += _

        plain_chars_in_data = self.get_occured(text_batch)
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        trsps = list(trsps)
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        plabels_cased, tdicts_cased = self.get_plabel_and_dict_core(trsps, trchs, False)
        plabels_uncased, tdicts_uncased = self.get_plabel_and_dict_core(trsps, trchs, True)
        # self.debug(trchs,"meow")
        return normprotos, [plabels_uncased, plabels_cased], [tdicts_uncased, tdicts_cased]


class NekoPrototypeSamplerFsl(NekoPrototypeSamplerBasic):
    def split(self, s):
        # not hurt if we have one more meme. In fact we need a random two-character non-sense
        return s.split("⑤⑨")

    def get_occured(self, text_batch):
        b = []
        for _ in text_batch: b += _.split("⑤⑨")
        return b

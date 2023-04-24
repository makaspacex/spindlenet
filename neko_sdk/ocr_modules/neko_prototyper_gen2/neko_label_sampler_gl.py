import random

import numpy as np
import torch

from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_abstractract_sampler import neko_prototype_sampler_static


class neko_prototype_sampler_gl(neko_prototype_sampler_static):
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
        protos = ((torch.cat(normprotos, dim=-1).squeeze(0).squeeze(0) + 1) * 127.5).detach().cpu().numpy().astype(
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
            trsps = [self.label_dict[self.EOS]]
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
        plabels, gplabels, tdicts, gtdicts = self.get_plabel_and_dictg(trsps, trchs)
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        # self.debug(trchs,"meow")
        return normprotos, plabels, gplabels, tdicts, gtdicts

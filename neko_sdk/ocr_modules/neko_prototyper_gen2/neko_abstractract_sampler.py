import copy

import numpy as np
import regex
import torch
from torch.nn import functional as trnf


class neko_prototype_sampler_static:

    def set_gbidict(self, oks):
        self.gbidict = {}
        gkset = {}
        gkcnt = 0
        for k in oks:
            drk = self.label_dict[k]
            if (drk not in gkset):
                gkset[drk] = gkcnt
                gkcnt += 1
            self.label_dict[k] = gkset[drk]
            self.gbidict[k] = gkset[drk]
            if (drk == self.label_dict[k]):
                self.gbidict[gkset[drk]] = k
            else:
                print("err")
                exit(9)

    def setup(self, meta):
        self.bidict = {}
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size, it's nightmare.
        list_character = list(meta["chars"])
        self.aligned_characters = meta["achars"]
        # characters without shape is generally what you do now want to sample.
        self.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        self.character = list(meta["sp_tokens"]) + list_character
        self.label_dict = meta["label_dict"]
        self.shaped_ids = set([self.label_dict[i] for i in self.shaped_characters])

        self.sp_cnt = len(meta["sp_tokens"])
        self.sp_tokens = meta["sp_tokens"]
        self.norm_protos = meta["protos"][self.sp_cnt:]

        unk = self.label_dict["[UNK]"]
        # if the dict does not provide an specific unk token, set it to -1
        for i, char in enumerate(self.character):
            # print(i, char)
            self.label_dict[char] = i
            self.bidict[char] = i
            self.bidict[i] = char

        # shapeless unk shall be excluded
        if (unk < 0):
            self.label_set = set(self.label_dict.values()) - {unk}
        else:
            self.label_set = set(self.label_dict.values())

        for i in range(len(self.norm_protos)):
            if self.norm_protos[i] is not None and self.norm_protos[i].max() > 20:
                self.norm_protos[i] = (self.norm_protos[i] - 127.5) / 128

        self.prototype_cnt = -1
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        self.masters = meta["master"]
        self.reduced_label_dict = {}
        self.reduced_bidict = {}

        kcnt = 0
        kset = {}
        ks = []
        ls = []
        for k in self.label_dict:
            ks.append(k)
            ls.append(self.label_dict[k])
        oks = [ks[i] for i in np.argsort(ls)]

        for k in oks:
            if (self.label_dict[k] in self.masters):
                drk = self.masters[self.label_dict[k]]
            else:
                drk = self.label_dict[k]
            if (drk not in kset):
                kset[drk] = kcnt
                kcnt += 1
            self.reduced_label_dict[k] = kset[drk]
            self.reduced_bidict[k] = kset[drk]
            if (drk == self.label_dict[k]):
                self.reduced_bidict[kset[drk]] = k

        self.set_gbidict(oks)

        # Foes includes the characters looks like each other
        # but never share labels (They may actually have linguistic relationships...
        # Like yanderes in a broken relationship[x]).
        # This set helps implement ohem like minibatching on the huge labelset.
        # e.g. 'u' and 'ü'
        self.foes = meta["foes"]
        self.servants = meta["servants"]
        # union set of friend, harem and foe.
        self.related_proto_ids = meta["relationships"]

    def setup_meta(self, meta_args):
        self.EOS = 0
        self.case_sensitive = meta_args["case_sensitive"]
        self.meta_args = meta_args
        self.masters_share = not self.case_sensitive
        if (meta_args["meta_path"] is None):
            return
        meta = torch.load(meta_args["meta_path"])

        self.setup(meta)

    def split(self, s):
        return regex.findall(r'\X', s, regex.U)

    def get_occured(self, text_batch):
        b = ""
        for _ in text_batch: b += _
        return list(set(regex.findall(r'\X', b, regex.U)))
        # every thing does not involve sampling

    def encode_fn_naive_noeos(self, tdict, label_batch):
        length = [len(self.split(s)) for s in label_batch]
        max_len = max(length)
        out = torch.zeros(len(label_batch), max_len).long()
        for i in range(0, len(label_batch)):
            cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                        for char in self.split(label_batch[i])])
            out[i][0:len(cur_encoded)] = cur_encoded
        return out, length

    def encode_noeos(self, proto, plabel, tdict, label_batch):
        if (not self.case_sensitive):
            label_batch = [l.lower() for l in label_batch]
        return self.encode_fn_naive_noeos(tdict, label_batch)

    # every thing does not involve sampling
    def encode_fn_naive(self, tdict, label_batch):
        max_len = max([len(self.split(s)) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len + 1).long() + self.EOS
        for i in range(0, len(label_batch)):
            cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                        for char in self.split(label_batch[i])])
            out[i][0:len(cur_encoded)] = cur_encoded
        return out

    def encode(self, proto, plabel, tdict, label_batch):
        if (not self.case_sensitive):
            label_batch = [l.lower() for l in label_batch]
        return self.encode_fn_naive(tdict, label_batch)

    def decode_train_noprob(self, net_out, length, protos, labels, tdict):
        # decoding prediction into text with geometric-mean probability
        # the probability is used to select the more realiable prediction when using bi-directional decoders
        out = []
        out_prob = []
        out_raw = []
        raw_out = net_out

        # net_out = trnf.softmax(net_out, dim=1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:,
                               0].tolist()
            current_text = ''.join([tdict[_] if _ in tdict else '' for _ in current_idx_list])

            out_raw.append(current_idx_list)
            out.append(current_text)
        return (out, out_raw)

    def decode(self, net_out, length, protos, labels, tdict):
        # decoding prediction into text with geometric-mean probability
        # the probability is used to select the more realiable prediction when using bi-directional decoders
        out = []
        out_prob = []
        out_raw = []
        raw_out = net_out

        net_out = trnf.softmax(net_out, dim=1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:,
                               0].tolist()
            current_text = ''.join([tdict[_] if _ in tdict else '' for _ in current_idx_list])

            current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]

            out_raw.append(list(raw_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:,
                                0].detach().cpu().numpy()))
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return (out, out_raw)

    def decode_beam(self, net_out, length, protos, labels, tdict, topk_ch=5, topk_beams=10):
        # text_d,pr_d=self.decode(net_out, length, protos, labels,tdict)

        out = []
        out_prob = []
        out_raw = []
        raw_out = net_out
        net_out = trnf.softmax(net_out, dim=1)
        beams = []
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(topk_ch)[
                1].tolist()
            current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(topk_ch)[
                0].cpu().numpy()
            active_lists = [[]]
            active_prob_lists = [[]]
            active_plist = [1]
            for timestamp in range(int(length[i].item())):
                new_list = []
                new_plist = []
                new_prob_lists = []
                aids = np.argsort(active_plist)[::-1][:topk_beams]
                for chid in range(topk_ch):
                    pr = current_probability[timestamp][chid]
                    ch = current_idx_list[timestamp][chid]
                    for aid in aids:
                        new_list.append(active_lists[aid] + [ch])
                        new_plist.append(pr * active_plist[aid])
                        new_prob_lists.append(active_prob_lists[aid] + [pr])
                active_lists = new_list
                active_plist = new_plist
                active_prob_lists = new_prob_lists
                pass
            aids = np.argsort(active_plist)[::-1][:topk_beams]
            current_idx_lists = [active_lists[aid] for aid in aids]
            current_probabilitys = [active_prob_lists[aid] for aid in aids]
            tdict[0] = "TOPNEP"
            current_texts = [''.join([tdict[_] if _ in tdict else '~' for _ in current_idx_list]) for current_idx_list
                             in current_idx_lists]
            current_texts = [t.split("TOPNEP")[0] for t in current_texts]

            # if(current_texts[0]!=text_d[i]):
            #     print("Oops")

            out_raw.append(list(raw_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:,
                                0].detach().cpu().numpy()))
            current_probability = 1  # torch.exp(torch.log(torch.tensor(current_probability)).sum() / current_probability.size[0])
            out.append(current_texts[0])
            out_prob.append(current_probability)
            beams.append(current_texts)
        return (out, out_raw, beams)
        pass

    def setup_sampler(self, sampler_args):
        pass

    def dump_all_impl(self, use_sp=True):
        if (use_sp):
            trsps = list(set([self.label_dict[i] for i in self.sp_tokens]))
        else:
            trsps = []
        trchs = list(set([self.label_dict[i] for i in self.shaped_characters]))
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return normprotos, plabels, tdicts

    def dump_all(self, metaargs=None, use_sp=True):
        if (metaargs is not None):
            if (metaargs["meta_path"] is None):
                return None, None, None
            a = copy.deepcopy(self)
            a.setup_meta(metaargs)
            return a.dump_all_impl(use_sp)
        return self.dump_all_impl(use_sp)

    def dump_allg_impl(self, use_sp=True):
        if (use_sp):
            trsps = list(set([self.label_dict[i] for i in self.sp_tokens]))
        else:
            trsps = []
        trchs = list(set([self.label_dict[i] for i in self.shaped_characters]))
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in trchs]
        plabels, gplabels, bidict, gbidict = self.get_plabel_and_dictg(trsps, trchs)
        return normprotos, plabels, gplabels, bidict, gbidict

    def dump_allg(self, metaargs=None, use_sp=True):
        if (metaargs is not None):
            a = copy.deepcopy(self)
            a.setup_meta(metaargs)
            return a.dump_allg_impl(use_sp)
        return self.dump_allg_impl(use_sp)

    def __init__(self, meta_args, sampler_args):
        self.setup_meta(meta_args)
        self.setup_sampler(sampler_args)

    def get_plabel_and_dict(self, sappids, normpids):
        return self.get_plabel_and_dict_core(sappids, normpids, self.masters_share)

    def get_plabel_and_dictg(self, sappids, normpids):
        return self.get_gplabel_and_dict_core(sappids, normpids, self.masters_share)

    # No semb shit here, semb comes form meta, not sampler
    def get_plabel_and_dict_core(self, sappids, normpids, masters_share):
        all_ids = sappids + normpids
        new_id = 0
        plabels = []
        labmap = {}
        bidict = {}
        gbidict = {}
        for i in all_ids:
            cha = self.aligned_characters[i]
            if (masters_share):
                vlab = self.masters[i]
            else:
                vlab = i
            if (vlab not in labmap):
                labmap[vlab] = new_id
                # A new label
                new_id += 1
                # sembs.append(self.semantic_embedding[vlab])
            alab = labmap[vlab]
            plabels.append(alab)
            bidict[alab] = cha
            bidict[cha] = alab

        plabels.append(new_id)
        bidict["[UNK]"] = new_id

        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        bidict[new_id] = "⑨"
        return torch.tensor(plabels), bidict

    # No semb shit here, semb comes form meta, not sampler
    def get_gplabel_and_dict_core(self, sappids, normpids, masters_share, use_sp=True):
        if (use_sp):
            all_ids = sappids + normpids
        else:
            all_ids = normpids
        new_id = 0
        plabels = []
        labmap = {}
        bidict = {}
        gplabels = []
        gmapping = []
        for i in all_ids:
            cha = self.aligned_characters[i]
            if (masters_share):
                vlab = self.masters[i]
            else:
                vlab = i
            vcha = self.aligned_characters[vlab]
            if (vlab not in labmap):
                labmap[vlab] = new_id
                # A new label
                new_id += 1
                # sembs.append(self.semantic_embedding[vlab])
            alab = labmap[vlab]
            plabels.append(alab)
            bidict[alab] = vcha
            bidict[cha] = alab
        plabels.append(new_id)
        bidict["[UNK]"] = new_id

        if (self.masters_share):
            gbidict = self.reduced_bidict
        else:
            gbidict = self.gbidict

        for i in range(new_id):
            gplabels.append(gbidict[bidict[i]])
        gplabels.append(gbidict["[UNK]"])
        gbidict[gbidict["[UNK]"]] = ""
        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        # set most special keys to "" if any.
        for s in sappids:
            bidict[s] = ""
        bidict[new_id] = "⑨"

        return torch.tensor(plabels), torch.tensor(gplabels), bidict, gbidict

# using mk5 DTD
import regex
import torch.nn.functional
import numpy as np

from neko_2020nocr.dan.common.common import flatten_label
from neko_2020nocr.dan.utils import LossCounter, NekoOsAttentionArCounter, NekoOswrAttentionArCounter
from neko_2021_mjt.modulars.neko_inflater import NekoInflater
from neko_2021_mjt.routines.neko_abstract_routines import NekoAbstractEvalRoutine
from neko_2021_mjt.routines.ocr_routines.mk5.osdan_routine_mk5 import NekoHdos2cRoutineCfmk5


# mk5 CF branch dropped predict-sample-predict support.

# A GP branch will be added if it's ever to be supported
# Mk7 CF branch uses CAM to perform length prediction, [s] is no more needed
class NekoHDOS2cRoutineCfmk7(NekoHdos2cRoutineCfmk5):

    def mk_proto(self, label, sampler, prototyper):
        normprotos, plabel, tdict = sampler.model.sample_charset_by_text(label, use_sp=False)
        # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8)
        # cv2.imshow("alphabets",im)
        # print([tdict[label.item()] for label in plabel])
        # cv2.waitKey(0)
        proto = prototyper(normprotos, use_sp=False)

        semb = None
        return proto, semb, plabel, tdict

    def fe_seq(self, clips, module_dict, length):
        seq = module_dict["seq"]
        features = module_dict["feature_extractor"](clips.cuda())
        features = [f.contiguous() for f in features]

        A, pred_length = module_dict["CAM"](features)
        out_emb = seq(features[-1], A, length)
        return out_emb, A, pred_length

    def fp_impl(self, input_dict, module_dict, logger_dict, nEpoch, batch_idx):
        label = input_dict["label"]
        clips = input_dict["image"]
        prototyper = module_dict["prototyper"]
        sampler = module_dict["sampler"]
        preds = module_dict["preds"]

        proto, semb, plabel, tdict = self.mk_proto(label, sampler, prototyper)
        target, length = sampler.model.encode_noeos(proto, plabel, tdict, label)
        label_flatten, length = flatten_label(target, EOSlen=0, length=length)
        target, label_flatten, culength = target.cuda(), label_flatten.cuda(), length.cuda().long()
        out_emb, A, pred_length = self.fe_seq(clips, module_dict, length)
        # net forward
        # Length dring training is known.
        fout_emb, _ = self.inflater.inflate(out_emb, length)
        lossess = []
        beams = []
        probs = []
        terms = []
        loss = torch.nn.functional.cross_entropy(pred_length, culength)
        for i in range(len(preds)):
            logits = preds[i](fout_emb, proto, plabel)
            choutput, prdt_prob = sampler.model.decode(logits, length, proto, plabel, tdict)
            loss_, terms_ = module_dict["losses"][i](proto, logits, label_flatten)
            loss = loss_ + loss
            beams.append(choutput)
            terms.append(terms_)

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))]
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss


class NekoHDOS2cRoutineCfmk7dt(NekoHDOS2cRoutineCfmk7):

    def fe_seq(self, clips, module_dict, length):
        seq = module_dict["seq"]
        features = module_dict["feature_extractor"](clips.cuda())
        features = [f.contiguous() for f in features]
        A, pred_length = module_dict["CAM"](features)
        out_emb = seq(features[-1], A, length)
        return out_emb, A, pred_length


class NekoHDOS2cRoutineCfmk7dtf(NekoHDOS2cRoutineCfmk7):

    def fe_seq(self, clips, module_dict, length):
        seq = module_dict["seq"]
        features = module_dict["feature_extractor"](clips.cuda())
        features = [f.contiguous() for f in features]
        featuresd = [f.detach() for f in features]
        A, pred_length = module_dict["CAM"](featuresd)
        out_emb = seq(features[-1], A, length)
        return out_emb, A, pred_length


class NekoHDOS2cEvalRoutineCfmk7(NekoAbstractEvalRoutine):

    def set_etc(self, args):
        self.maxT = args["maxT"]
        self.inflater = NekoInflater()

    def set_loggers(self, log_path, name, args):
        try:
            if (args["measure_rej"] == True):
                self.logger_dict = {"accr": NekoOswrAttentionArCounter("[" + name + "]" + "test_accr", False),
                                    }
            else:
                self.logger_dict = {
                    "accr": NekoOsAttentionArCounter("[" + name + "]" + "test_accr", False),
                    "loss": LossCounter("[" + name + "]" + "train_accr"),
                }
        except:
            self.logger_dict = {
                "accr": NekoOsAttentionArCounter("[" + name + "]" + "test_accr", False),
                "loss": LossCounter("[" + name + "]" + "train_accr"),
            }

    def test_impl(self, data_dict, module_dict, logger_dict):
        data, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"]
        preds = module_dict["preds"]
        seq = module_dict["seq"]
        sampler = module_dict["sampler"]

        data = data.cuda()
        features = module_dict["feature_extractor"](data)
        A, pred_length = module_dict["CAM"](features)
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone()
        out_emb = seq(features[-1], A, None)
        # lossess = []
        beams_ = []
        probs = []
        # terms = []
        logits = []
        loss = 0
        
        for i in range(len(preds)):
            nT, nB = out_emb.shape[0], out_emb.shape[1]
            _logits = preds[i](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1])
            
            
            logits.append(np.transpose(_logits.cpu().numpy(),axes=(1,0,2)))
            _logits, _ = self.inflater.inflate(_logits, pred_length)
            
            choutput, prdt_prob = sampler.model.decode(_logits, pred_length, proto, plabel, tdict)
            beams_.append(choutput)
            probs.append(prdt_prob)
            # loss_, terms_ = module_dict["losses"][i](proto, preds, label_flatten)
            # loss = loss_ + loss
            beams_.append(choutput)
            # terms.append(terms_)
        beams = []
        for i in range(features[-1].shape[0]):
            beam = []
            for j in range(len(beams_)):
                beam.append(beams_[j][i])
            beams.append(beam)
        # A=A.max(dim=2)[0]
        flabel = []
        for l in label:
            s = ""
            for c in regex.findall(r'\X', l, regex.U):
                if (c not in tdict):
                    s += "⑨"
                else:
                    s += c
            flabel.append(s)
        logger_dict["accr"].add_iter(beams_[0], pred_length, flabel)
        rdict = {}
        
        statis  = {}
        # statis['out_emb'] = np.transpose(out_emb.cpu().numpy(),axes=(1,0,2))
        statis['logits'] = logits
        statis['label'] = label
        
        return beams_[0], rdict, beams,statis

        # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)

    def vis_logits_impl(self, img, data_dict, module_dict, at_time):

        _, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"]
        data = img
        preds = module_dict["preds"]
        seq = module_dict["seq"]
        sampler = module_dict["sampler"]

        data = data.cuda()
        data = torch.nn.Parameter(data, requires_grad=True)

        features = module_dict["feature_extractor"](data)
        A, pred_length = module_dict["CAM"](features)
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone()
        out_emb = seq(features[-1], A, None)
        # lossess = []
        beams_ = []
        probs = []
        # terms = []

        loss = 0
        nT, nB = out_emb.shape[0], out_emb.shape[1]
        logits = preds[0](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1])
        logits, _ = self.inflater.inflate(logits, pred_length)

        if (len(logits) <= at_time):
            return None
        return logits[at_time]

    def pretest_impl(self, modular_dict, metaargs, **kwargs):
        rot = kwargs["rot"]
        normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs, use_sp=False)
        if (not rot):
            proto = modular_dict["prototyper"](normproto, use_sp=False)
        else:
            proto = modular_dict["prototyper"](normproto, rot)
        return {"proto": proto, "plabel": plabel, "tdict": tdict}

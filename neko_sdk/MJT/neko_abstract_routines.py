import torch
import torch.nn.parallel


class neko_abstract_routine:

    def set_loggers(self, log_path, log_each, name):
        self.logger_dict = {}
        pass

    def set_etc(self, args):
        pass

    def __init__(self, args):
        mod_cvt_dicts, inp_cvt_dicts = \
            args["mod_cvt_dicts"], args["inp_cvt_dicts"]
        self.set_etc(args)
        self.name = args["name"]
        log_path, log_each = \
            args["log_path"], args["log_each"]
        self.log_each = log_each

        # tells which module is which in modular_dict.
        # we may have two identical routines(DAN_char and DAN_word sharing only part of modules)
        self.mod_cvt_dict = mod_cvt_dicts
        self.inp_cvt_dict = inp_cvt_dicts
        self.set_loggers(log_path, log_each, args["name"])
        pass

    def log_show(self):
        for k in self.logger_dict:
            self.logger_dict[k].show()

    def grab_nested(self, moduleterm, modular_dict):
        if (type(moduleterm) is list):
            return [self.grab_nested(n, modular_dict) for n in moduleterm]
        else:
            return modular_dict[moduleterm]

    def grab_modules(self, input_dict, modular_dict):
        mdict = {}
        idict = {}
        for k in self.mod_cvt_dict:
            mdict[k] = self.grab_nested(self.mod_cvt_dict[k], modular_dict)
        for k in self.inp_cvt_dict:
            idict[k] = input_dict[self.inp_cvt_dict[k]]
        return idict, mdict

    def fp_impl(self, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        return torch.tensor(0)

    def bp_impl(self, loss):
        loss.backward()

    def fpbp_impl(self, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        loss = self.fp_impl(input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device)
        self.bp_impl(loss)
        pass

    def fpbp_amp_impl(self, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        with torch.cuda.amp.autocast():
            loss = self.fp_impl(input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device)
        self.bp_impl(loss)
        pass

    def fp(self, input_dict, modular_dict, nEpoch, batch_idx, device):
        idict, mdict = self.grab_modules(input_dict, modular_dict)
        loss = self.fp_impl(idict, mdict, self.logger_dict, nEpoch, batch_idx, device)
        if (batch_idx % self.log_each == 0):
            self.log_show()
        return loss

    def fpbp(self, input_dict, modular_dict, nEpoch, batch_idx, device="cuda"):
        idict, mdict = self.grab_modules(input_dict, modular_dict)
        if ("debug_path" in input_dict):
            idict["debug_path"] = input_dict["debug_path"]
        if ("vdbg" in input_dict):
            idict["vdbg"] = input_dict["vdbg"]

        ret = self.fpbp_impl(idict, mdict, self.logger_dict, nEpoch, batch_idx, device)
        if (batch_idx % self.log_each == 0):
            self.log_show()
        return ret

    def fpbp_amp(self, input_dict, modular_dict, nEpoch, batch_idx, device="cuda"):
        idict, mdict = self.grab_modules(input_dict, modular_dict)
        ret = self.fpbp_amp_impl(idict, mdict, self.logger_dict, nEpoch, batch_idx, device)
        if (batch_idx % self.log_each == 0):
            self.log_show()
        return ret

    def __call__(self, input_dict, modular_dict, nEpoch, batch_idx):
        self.fpbp(input_dict, modular_dict, nEpoch, batch_idx)
        return None


# you may or may not sharing configs with training.
class neko_abstract_eval_routine:
    def clear_loggers(self):
        for l in self.logger_dict:
            self.logger_dict[l].clear()

    def set_loggers(self, log_path, name, args):
        self.logger_dict = {}

    def set_etc(self, args):
        pass

    def show_log(self):
        for lk in self.logger_dict:
            self.logger_dict[lk].show()

    def ret_log(self):
        ret = {}
        for lk in self.logger_dict:
            ret[lk] = self.logger_dict[lk].show()
            self.logger_dict[lk].show()
        return ret

    def __init__(self, args):
        self.set_etc(args)

        mod_cvt_dicts, inp_cvt_dicts = \
            args["mod_cvt_dicts"], args["inp_cvt_dicts"]
        log_path = args["log_path"]
        # tells which module is which in modular_dict.
        # we may have two identical routines(DAN_char and DAN_word sharing only part of modules)
        self.mod_cvt_dict = mod_cvt_dicts
        self.inp_cvt_dict = inp_cvt_dicts
        self.set_loggers(log_path, args["name"], args)
        pass

    def interpret_mods(self, modular_dict):
        mdict = {}
        for k in self.mod_cvt_dict:
            if (type(self.mod_cvt_dict[k]) is list):
                mdict[k] = []
                for n in self.mod_cvt_dict[k]:
                    # a weird string to ensure missing model is intentional
                    mdict[k].append(modular_dict[n])
            else:
                if (self.mod_cvt_dict[k] == "NEP_skipped_NEP"):
                    mdict[k] = None
                else:
                    mdict[k] = modular_dict[self.mod_cvt_dict[k]]
        return mdict

    def grab_modules(self, input_dict, modular_dict):
        idict = {}
        mdict = self.interpret_mods(modular_dict)
        for k in self.inp_cvt_dict:
            idict[k] = input_dict[self.inp_cvt_dict[k]]
        return idict, mdict


w


def pretest_impl(self, modular_dict, metaargs, **kwargs):
    rot = kwargs["rot"]
    normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs)
    if ("[s]" in tdict):
        tdict[tdict["[s]"]] = 0
    if (not rot):
        proto = modular_dict["prototyper"](normproto)
    else:
        proto = modular_dict["prototyper"](normproto, rot)
    return {"proto": proto, "plabel": plabel, "tdict": tdict}


def pretest(self, modular_dict, metaargs, override=None, **kwargs):
    mdict = self.interpret_mods(modular_dict)
    if (override is not None):
        for i in override:
            mdict[i] = modular_dict[i]
    return self.pretest_impl(mdict, metaargs, **kwargs)


def test_impl(self, input_dict, modular_dict, logger_dict):
    pass


def test_topk_impl(self, input_dict, modular_dict, logger_dict, k):
    pass


# return logits
def vis_logits_impl(self, img, data_dict, modular_dict, at_time):
    pass


def vis_logit(self, img, input_dict, modular_dict, at_time, override=None, vdbg=None):
    idict, mdict = self.grab_modules(input_dict, modular_dict)
    if (not (vdbg is None)):
        idict["vdbg"] = vdbg
    if (override is not None):
        for i in override:
            mdict[i] = modular_dict[i]
    return self.vis_logits_impl(img, idict, mdict, at_time)


def test(self, input_dict, modular_dict, override=None, vdbg=None):
    idict, mdict = self.grab_modules(input_dict, modular_dict)
    if (not (vdbg is None)):
        idict["vdbg"] = vdbg
        try:
            idict["idi"] = input_dict["idi"]
        except:
            pass
    if (override is not None):
        for i in override:
            mdict[i] = modular_dict[i]
    return self.test_impl(idict, mdict, self.logger_dict)


def test_topk(self, input_dict, modular_dict, k, override=None, vdbg=None):
    idict, mdict = self.grab_modules(input_dict, modular_dict)
    if (not (vdbg is None)):
        idict["vdbg"] = vdbg
        try:
            idict["idi"] = input_dict["idi"]
        except:
            pass
    if (override is not None):
        for i in override:
            mdict[i] = modular_dict[i]
    return self.test_topk_impl(idict, mdict, self.logger_dict, k)

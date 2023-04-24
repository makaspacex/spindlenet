import torch

from neko_2021_mjt.debug_and_visualize.laserbeam import NekoBackprop


# we need something to wrap the routine.

class Bogomod(torch.nn.Module):
    def __init__(self, visible, moddict):
        super(Bogomod, self).__init__()
        self.core = visible
        mcvt = self.core.mod_cvt_dict
        tmoddict = {}
        for name in mcvt:
            if (type(mcvt[name]) == list or mcvt[name] == "NEPnoneNEP"):
                continue
            tm = moddict[mcvt[name]].get_torch_module_dict()
            if (tm is None):
                continue
            tmoddict[name] = tm
        for name in tmoddict:
            self.add_module(name, tmoddict[name])

    def load(self, input_dict, modular_dict, at_time, bid=0):
        self.t = at_time
        self.modular_dict = modular_dict
        self.input_dict = input_dict
        image = input_dict["image"][bid:bid + 1]
        text = self.input_dict["label"][bid][at_time]
        if (text in input_dict["tdict"]):
            raberu = input_dict["tdict"][text]
        else:
            raberu = input_dict["tdict"]["[UNK]"]  # Umm, just to distinguish it from ascii encoding
        return image, text, raberu

    def forward(self, image):
        logit = self.core.vis_logit(image, self.input_dict, self.modular_dict, self.t)
        if (logit is None):
            return None
        return logit.unsqueeze(0)


# cats LOVE laser dots.
class NekoLaser(object):
    def __init__(self, model, moddict):
        self.model = Bogomod(model, moddict)
        self.bper = NekoBackprop(self.model, self.model.feature_extractor.shared_fe_0_conv, 0)

    def vis_chars(self, input_dict, modular_dict):
        bs = input_dict["image"].shape[0]
        grads = []
        for bid in range(bs):
            text = input_dict["label"][bid]
            cgs = []
            for t in range(0, len(text)):
                image, text, raberu = self.model.load(input_dict, modular_dict, t, bid)
                grad_t = self.bper.calculate_gradients(image, raberu, take_max=True, guided=False, use_gpu=True)
                if (grad_t is None):
                    continue
                cgs.append(grad_t)
            grads.append(cgs)
        return grads

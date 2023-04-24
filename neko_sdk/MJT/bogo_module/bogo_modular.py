from neko_sdk.MJT.bogo_module.servant_module import neko_stand_basic


class neko_bogo_modular:
    def __init__(self, forwardable):
        # you should never have a
        self.model = forwardable
        self.save_each = -9

    # forked stands can only do forward. Fancy controls can yield weird race conditions.

    def train(self, training=True):
        pass

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def detach(self):
        self.model.detach()

    def attach(self):
        self.model.attach()

    def dnattach_fwd(self, *args, **kwargs):
        self.detach()
        ret = self(*args, **kwargs)
        self.attach()
        return ret

    def replicate(self, devices):
        stands = self.model.replicate(devices)
        self.stands = [neko_stand_basic(stand) for stand in stands]
        return self.stands

    def get_torch_modular_dict(self):
        try:
            return self.model.get_torch_modular_dict()
        except:
            return None

    # does nothing
    def eval(self):
        pass

    def normgrad(self):
        pass

    def zero_grad(self):
        pass

    def load(self, itrkey):
        pass

    def cuda(self):
        pass

    def save(self, nEpoch):
        pass

    def save_if_needed(self, nEpoch, batch_idx):
        pass

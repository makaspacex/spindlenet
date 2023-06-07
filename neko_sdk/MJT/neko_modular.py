import torch
from torch import nn
from torch.nn import parallel as trnp

from neko_sdk.MJT.bogo_module.servant_module import NekoStandBasic


class NekoModular:
    def __init__(self, path, name, module, save_each=20000):
        self.path = path
        self.model = module
        self.name = name
        self.save_each = save_each
        self.stands = None

    def get_torch_modular_dict(self):
        if (isinstance(self.model, nn.Module)):
            return self.model
        else:
            return None

    def replicate(self, devices):
        self.model.to(devices[0])
        models = trnp.replicate(self.model, devices)
        self.stands = [NekoStandBasic(model) for model in models]
        return self.stands

    def detach(self):
        self.model.requires_grad_(False)

    def attach(self):
        self.model.requires_grad_(True)

    def train(self, training=True):
        self.model.train(training)

    def eval(self):
        self.model.eval()

    def normgrad(self):
        if self.save_each > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 20, 2)

    def cuda(self):
        self.model.cuda()

    def zero_grad(self):
        if self.save_each > 0:
            for param in self.model.parameters():
                param.grad = None

            if (self.stands is not None):
                for stand in self.stands:
                    stand.model.zero_grad()

    def load(self, itrkey):
        p = self.path + itrkey + ".pth"
        try:
            self.model.load_state_dict(torch.load(p).state_dict())
        except:
            try:
                self.model.load_state_dict(torch.load(p))
                print(self.name, "loaded as a hack")
            except:
                print(self.name, "cannot load", "itr", p, ", starting fresh")

    def save(self, nEpoch):
        if (self.save_each > 0):
            torch.save(self.model, self.path + '_E{}.pth'.format(nEpoch))
            torch.save(self.model, self.path + 'latest.pth')

    def save_if_needed(self, nEpoch, batch_idx):
        if (self.save_each > 0 and batch_idx % self.save_each == 0):
            print("Saving", self.path + '_E{}_I{}.pth'.format(nEpoch, batch_idx))
            torch.save(self.model, self.path + '_E{}_I{}.pth'.format(nEpoch, batch_idx))
            torch.save(self.model, self.path + 'latest.pth')

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

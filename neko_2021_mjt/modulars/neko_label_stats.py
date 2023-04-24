import torch
from torch import nn


class NekoPystat(nn.Module):
    def __init__(self, max_capacity=900000):
        super(NekoPystat, self).__init__()
        self.cnts = torch.nn.Parameter(torch.zeros(max_capacity), requires_grad=False)
        self.total = torch.nn.Parameter(torch.tensor(1e-9), requires_grad=False)
        self.lclipping_freq = 0.01
        self.cdict = {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "cdict"] = self.cdict
        super(NekoPystat, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.cdict = state_dict[prefix + "cdict"]
        del state_dict[prefix + "cdict"]
        super(NekoPystat, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                      missing_keys, unexpected_keys, error_msgs)

    def forward_train(self, flatten_label, gdict, llen):
        for i in range(flatten_label.shape[0]):
            ch = gdict[flatten_label[i].item()]
            if (ch not in self.cdict):
                self.cdict[ch] = len(self.cdict)
            self.cnts[self.cdict[ch]] += 1
            self.total += 1
        return self.forward_eval(gdict, llen)

    def forward_eval(self, gdict, llen):
        ret = torch.zeros(llen, dtype=torch.float, device=self.cnts.device)
        for i in range(llen):
            ch = gdict[i]
            if (ch not in self.cdict):
                self.cdict[ch] = len(self.cdict)
            ret[i] = self.cnts[self.cdict[ch]]
        return torch.clip(ret / self.total, self.lclipping_freq)

    def forward(self, gdict, flatten_label, llen):
        # floats rounds at 16,777,216, we assume the estimation is good enough when it saw ~16M characters.
        if (self.training and self.total < 16777009):
            return self.forward_train(flatten_label, gdict, llen)
        else:
            return self.forward_eval(gdict, llen)


if __name__ == '__main__':
    a = NekoPystat(9)
    a.cdict["a"] = 9
    torch.save(a.state_dict(), "test.pt")
    b = NekoPystat(9)
    b.load_state_dict(torch.load("test.pt"))
    print(b.cdict["a"])

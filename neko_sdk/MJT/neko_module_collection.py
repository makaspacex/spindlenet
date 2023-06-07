from torch import nn


class NekoModuleCollection(nn.Module):
    def __init__(self, **kwargs):
        super(NekoModuleCollection, self).__init__()
        self.build_layers(**kwargs)

    def __getitem__(self, item):
        return self._modules[item]

    def build_layers(self, **kwargs):
        pass

    def setup_modules_core(self, mdict, prefix):
        name_dict = {}
        for k in mdict:
            if (type(mdict[k]) is dict):
                subname_dict = self.setup_modules_core(mdict[k], prefix + "_" + k)
                name_dict[k] = subname_dict
            else:
                self.add_module(prefix + "_" + k, mdict[k])
                name_dict[k] = prefix + "_" + k
        return name_dict

    def setup_modules(self, mdict, prefix):
        self.name_dict = self.setup_modules_core(mdict, prefix)

    def forward(self, input, debug=False):
        # This won't work as this is just a holder
        exit(9)

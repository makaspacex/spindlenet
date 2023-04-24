import time

import torch
from torch.utils.data.dataloader import DataLoader


class NekoJointLoader:
    def __init__(self, dataloadercfgs, length):
        self.dataloaders = []
        self.ddict = {}
        self.names = []
        i = 0
        setcfgs = dataloadercfgs["subsets"]
        for name in setcfgs:
            self.ddict[name] = i
            i += 1
            self.names.append(name)
            cfg = setcfgs[name]
            train_data_set = cfg['type'](**cfg['ds_args'])
            train_loader = DataLoader(train_data_set, **cfg['dl_args'])
            self.dataloaders.append(train_loader)
        self.iters = [iter(loader) for loader in self.dataloaders]
        self.length = length

    def next(self):
        ret = {}
        for name in self.names:
            id = self.ddict[name]
            try:
                # use some nonsense that always trigger the reset loader behaviour
                # nep[1000]=1
                rett = self.iters[id].__next__()
            except:
                a = self.iters[id]
                self.iters[id] = None
                del a
                time.sleep(2)  # Prevent possible deadlock during epoch transition
                self.iters[id] = iter(self.dataloaders[id])
                rett = self.iters[id].__next__()

            for t in rett:
                if (torch.is_tensor(rett[t])):
                    ret[name + "_" + t] = rett[t].contiguous()
                else:
                    ret[name + "_" + t] = rett[t]
        return ret

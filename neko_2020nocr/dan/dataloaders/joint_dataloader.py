from torch.utils.data import DataLoader

from neko_sdk.dataloaders.neko_vlwrapper import NekoVlwrapper


# dataloader like, but capable for multitasking.
class NekoJointDataLoader:
    def __init__(self, subsets):
        self.loaders = []
        self.maxlen = 0
        for subset_name in subsets:
            dscfg = subsets[subset_name]["dscfg"]
            dataset = subsets[subset_name]["dstype"](**dscfg)
            ldrcfg = subsets[subset_name]["loadercfg"]
            l = DataLoader(dataset, **ldrcfg)
            self.loaders.append(NekoVlwrapper(l))
            length = len(l)
            if (length > self.maxlen):
                self.maxlen = length

    def get_batch(self):
        return [l.get_batch() for l in self.loaders]

    def __len__(self):
        return self.maxlen


#
# class neko_joint_data_loader:
#
#     def __init__(self, datasets, batch_sizes, shuffle=False, sampler=None,
#                  batch_sampler=None, num_workers=0, collate_fn=None,
#                  pin_memory=False, drop_last=False, timeout=0,
#                  worker_init_fn=None, multiprocessing_context=None,
#                  generator=None):
#         self.loaders=[]
#         self.maxlen=0
#         if(collate_fn is None):
#             collate_fn=[None for _ in datasets]
#         for d,b,c in zip(datasets,batch_sizes,collate_fn):
#             l=DataLoader(d,b,shuffle,sampler,
#                            batch_sampler,num_workers,c,
#                            pin_memory,drop_last,timeout,worker_init_fn,
#                            multiprocessing_context,generator)
#             self.loaders.append(neko_vlwrapper(l))
#             length=len(l)
#             if(length>self.maxlen):
#                 self.maxlen=length
#     def __iter__(self):
#         return [l.get_batch() for l in self.loaders]
#
#     def __len__(self):
#         return self.maxlen
def neko_joint_data_loader_getter(subsets):
    loaders = []

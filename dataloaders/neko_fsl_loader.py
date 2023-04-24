import os.path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# returns raw image,rectified_image and so on.
# let's make this a small 2-phasemodel
from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt


# we have a few anchors. The model selects on aspect ratios, and the model selects for sort edge size.

class NekoFslTaskLoaderNoProto(object):
    def __init__(self, db_root, waycnt, ucnt, labelset=None, dsize=[32, 32]):
        meta = torch.load(os.path.join(db_root, "meta.pt"))
        self.meta = {}
        self.dsize = tuple(dsize)
        self.keyz = []
        self.waycnt = waycnt
        self.ucnt = ucnt
        # some "characters" can be stupidly loong.
        self.db = neko_ocr_lmdb_mgmt(db_root, True, 20)
        for c in meta:
            if (labelset is not None):
                if (c in labelset):
                    self.meta[c] = meta[c]
                    self.keyz.append(c)
            else:
                self.meta[c] = meta[c]
                self.keyz.append(c)

    def get_random(self, lab, shot):
        name = self.meta[lab]
        images = []
        items = random.choices(name, k=shot)
        for item in items:
            image = self.db.get_encoded_im_by_name(item)
            images.append(cv2.resize(np.array(image), self.dsize))
        return images

    def batch_labelset(self, shots):
        assert (shots > 1);  # Do not be a jerk like Erdogan...
        if (len(self.keyz) > self.waycnt + self.ucnt):
            blab = random.sample(self.keyz, self.waycnt + self.ucnt)
        else:
            blab = list(self.keyz)
            random.shuffle(blab)
        samples = []
        usamples = []
        ulabels = []
        labels = []
        sc = min(self.waycnt, len(blab))
        tdict = {sc: "[-]", "[UNK]": sc}
        for i in range(sc):
            lab = blab[i]
            samples += self.get_random(lab, shots)
            labels += ([i for _ in range(shots)])
            tdict[lab] = i
            tdict[i] = lab
        if (sc < len(blab)):
            ulabels += [sc for _ in range(len(blab) - sc)]
            for i in range(len(blab) - sc):
                lab = blab[i + sc]
                usamples += self.get_random(lab, 1)

        # you can, definitely, connect usamplesto samples.
        # However, this trick allows you to take more control.
        return {
            "samples": torch.tensor(np.array(samples)).permute(0, 3, 1, 2) / 255.,
            "labels": torch.tensor(labels),
            "usamples": torch.tensor(usamples).permute(0, 3, 1, 2) / 255.,
            "ulabels": torch.tensor(ulabels),
            "tdict": tdict,
        }


class NekoFslNoProtoTaskDataset(Dataset):
    def __init__(self, db_root, waycnt, ucnt, labelset=None, dsize=[32, 32], shots=2, vlen=2000):
        self.shots = shots
        self.vlen = vlen

        self.core = NekoFslTaskLoaderNoProto(db_root, waycnt, ucnt, labelset, dsize)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return self.core.batch_labelset(self.shots)


def show_batch(sample, label, tdict):
    sp = (sample.permute(0, 2, 3, 1)) * 255
    mps = []
    sps = []
    pps = []
    for i in range(len(sp)):
        sps.append(sp[i])

    v = torch.cat([torch.cat(sps, 1)], 0)
    v = v.detach().cpu().numpy()[:, :, ::-1].astype(np.uint8)

    print([tdict[i.item()] for i in label])
    cv2.namedWindow("w", 0)
    cv2.imshow("w", v)
    cv2.waitKey(0)


def collate_fn(d):
    return d[0]


from dataloaders.sampler import RandomSampler

if __name__ == '__main__':
    l = NekoFslNoProtoTaskDataset("/home/lasercat/ssddata/OBC306/", 32, 4)
    dl = DataLoader(l, collate_fn=collate_fn, num_workers=3, sampler=RandomSampler(None))
    for d in dl:
        show_batch(d["samples"], d["labels"], d["tdict"])
        pass

#
# if __name__ == '__main__':
#     #
#
#     l=character_proto_task_loader("/home/lasercat/ssddata/charset_lmdb/","","/home/lasercat/ssddata/synth_data/bgim",16,32,4)
#     for i in range(19):
#         d=l.batch_charset(1)
#
#         pass

import os.path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# returns raw image,rectified_image and so on.
# let's make this a small 2-phasemodel
from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import NekoOcrLmdbMgmt


# we have a few anchors. The model selects on aspect ratios, and the model selects for sort edge size.

class NekoFslTaskLoaderNoProtoGen2(object):
    def __init__(self, db_root, waycnt, ucnt, labelset=None, dsize=[32, 32]):
        meta = torch.load(os.path.join(db_root, "meta.pt"))
        self.meta = {}
        self.qhbaug = True
        self.dsize = tuple(dsize)
        self.keyz = []
        self.keyz_dict = {}
        self.waycnt = waycnt
        self.ucnt = ucnt
        # some "characters" can be stupidly loong.
        self.db = NekoOcrLmdbMgmt(db_root, True, 20)
        glcnt = 0
        for c in meta:
            if (labelset is not None):
                if (c in labelset):
                    self.meta[c] = meta[c]
                    self.keyz.append(c)
                    self.keyz_dict[c] = glcnt
                    glcnt += 1
            else:
                self.meta[c] = meta[c]
                self.keyz.append(c)
                self.keyz_dict[c] = glcnt
                glcnt += 1

    def get_random(self, lab, shot):
        name = self.meta[lab]
        images = []
        if (len(name) > shot):
            items = random.sample(name, k=shot)
        else:
            items = random.choices(name, k=shot)
        for item in items:
            image = self.db.get_encoded_im_by_name(item)
            image = np.array(image)
            # if(self.qhbaug):
            #     image=qhbwarp(np.array(image),10)
            images.append(cv2.resize(image, self.dsize))
        return images

    def batch_labelset(self, ksupport, kquery):
        if (len(self.keyz) > self.waycnt + self.ucnt):
            blab = random.sample(self.keyz, self.waycnt + self.ucnt)
        else:
            blab = list(self.keyz)
            random.shuffle(blab)
        supports = []
        queries = []
        slabels = []
        qlabels = []
        sglabels = []
        qglabels = []
        sc = min(self.waycnt, len(blab))
        tdict = {sc: "[-]", "[UNK]": sc}
        for i in range(sc):
            lab = blab[i]
            clssamples = self.get_random(lab, ksupport + kquery)
            labeles = [i for _ in range(ksupport + kquery)]
            # for support, we do not have sample specific semantic, for now.
            # But it is not really the case (semantic is not single centered Guassian,
            # and centers can be correlated to different visual centers as well.)
            # i.e. font face is highly correlated to scenarios, which has a strong
            # affection on the content hence context.
            sglabels.append(self.keyz_dict[lab])
            qglabels += [self.keyz_dict[lab] for _ in range(kquery)]
            supports += clssamples[:ksupport]
            slabels += labeles[:ksupport]
            queries += clssamples[ksupport:]
            qlabels += labeles[ksupport:]
            tdict[lab] = i
            tdict[i] = lab
        slabels.append(tdict["[UNK]"])
        if (sc < len(blab)):
            qlabels += [sc for _ in range(len(blab) - sc)]
            for i in range(len(blab) - sc):
                lab = blab[i + sc]
                qglabels.append(self.keyz_dict[lab])
                queries += self.get_random(lab, 1)

        # you can, definitely, connect usamplesto samples.
        # However, this trick allows you to take more control.
        return {
            "samples": torch.tensor(queries).permute(0, 3, 1, 2) / 127.5 - 1,
            "labels": torch.tensor(qlabels),
            "protos": torch.tensor(supports).permute(0, 3, 1, 2) / 127.5 - 1,
            "plabels": torch.tensor(slabels),
            "qglabels": torch.tensor(qglabels),  # if you opt to use all semantic protos.
            "sglabels": torch.tensor(sglabels),
            # if you opt to use semantic protos like protos. You need to sample the semantic protos with sglables.
            "tdict": tdict,

            # "smsk": torch.tensor(sample_masks).permute(0, 3, 1, 2) / 255,
        }


class NekoFslNoProtoTaskDatasetg2(Dataset):
    def __init__(self, db_root, waycnt, ucnt, labelset=None, dsize=[32, 32], shots=2, vlen=2000):
        self.shots = shots
        self.vlen = vlen

        self.core = NekoFslTaskLoaderNoProtoGen2(db_root, waycnt, ucnt, labelset, dsize)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return self.core.batch_labelset(1, self.shots)


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


if __name__ == '__main__':
    pass
    # l = neko_fsl_no_proto_task_dataset("/home/lasercat/ssddata/OBC306/", 32,4)
    # dl=DataLoader(l,collate_fn=collate_fn,num_workers=3,sampler=randomsampler(None))
    # for d in dl:
    #     show_batch(d["samples"], d["labels"],d["tdict"])
    #     pass

#
# if __name__ == '__main__':
#     #
#
#     l=character_proto_task_loader("/home/lasercat/ssddata/charset_lmdb/","","/home/lasercat/ssddata/synth_data/bgim",16,32,4)
#     for i in range(19):
#         d=l.batch_charset(1)
#
#         pass

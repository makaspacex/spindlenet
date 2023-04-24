import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# returns raw image,rectified_image and so on.
# let's make this a small 2-phasemodel
from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt


# we have a few anchors. The model selects on aspect ratios, and the model selects for sort edge size.


class neko_gfsl_eval_task_dataset(Dataset):
    def __init__(self, db_root, labelset=None, dsize=[32, 32]):
        meta = torch.load(os.path.join(db_root, "meta.pt"))
        self.samples = []
        self.dsize = tuple(dsize)
        # some "characters" can be stupidly loong.
        self.db = neko_ocr_lmdb_mgmt(db_root, True, 20)
        for c in meta:
            if (labelset is not None):
                if (c in labelset):
                    for s in meta[c]:
                        self.samples.append([s, c])
            else:
                for s in meta[c]:
                    self.samples.append([s, c])
        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        iname, lab = self.samples[idx]
        image = self.db.get_encoded_im_by_name(iname)
        image = cv2.resize(np.array(image), self.dsize)
        image = torch.tensor(image).permute(2, 0, 1) / 255.,
        return {"samples": image, "labels": lab}


def show_batch(sample, label):
    sp = (torch.cat(sample).permute(0, 2, 3, 1)) * 255
    mps = []
    sps = []
    pps = []
    for i in range(len(sp)):
        sps.append(sp[i])

    v = torch.cat([torch.cat(sps, 1)], 0)
    v = v.detach().cpu().numpy()[:, :, ::-1].astype(np.uint8)

    print([i for i in label])
    cv2.namedWindow("w", 0)
    cv2.imshow("w", v)
    cv2.waitKey(0)


if __name__ == '__main__':
    l = neko_gfsl_eval_task_dataset("/home/lasercat/ssddata/jrc_tt100k_mmcls_train/")
    dl = DataLoader(l, 13, num_workers=0)
    for d in dl:
        show_batch(d["samples"], d["labels"])
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

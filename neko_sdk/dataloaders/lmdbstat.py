# coding:utf-8
import random
import sys

import lmdb
import numpy as np
import six
from PIL import Image
from torch.utils.data import Dataset

from neko_sdk.ocr_modules.io.data_tiding import neko_DAN_padding


class lmdbDataset(Dataset):
    def init_etc(self):
        pass

    def set_dss(self, roots):
        for i in range(0, len(roots)):
            env = lmdb.open(
                roots[i],
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            if not env:
                print('cannot creat lmdb from %s' % (roots[i]))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.roots.append(roots[i])
            self.envs.append(env)
            self.init_etc()

    def __init__(self, roots=None, ratio=None, img_height=32, img_width=128,
                 transform=None, global_state='Test', maxT=25, repeat=1, qhb_aug=False, force_target_ratio=None,
                 novert=True):
        self.envs = []
        self.roots = []
        self.maxT = maxT
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.global_state = global_state
        self.repeat = repeat
        self.qhb_aug = qhb_aug
        self.set_dss(roots)
        self.novert = novert
        if ratio != None:
            assert len(roots) == len(ratio), 'length of ratio must equal to length of roots!'
            for i in range(0, len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0, len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))

        self.transform = transform
        self.maxlen = max(self.lengths)
        self.img_height = img_height
        self.img_width = img_width
        # Issue12
        if (force_target_ratio is None):
            try:
                self.target_ratio = img_width / float(img_height)
            except:
                print("failed setting target_ration")
        else:
            self.target_ratio = force_target_ratio

    def __fromwhich__(self):
        rd = random.random()
        total = 0
        for i in range(0, len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img):
        img, bmask = neko_DAN_padding(img, None,
                                      img_width=self.img_width,
                                      img_height=self.img_height,
                                      target_ratio=self.target_ratio,
                                      qhb_aug=self.qhb_aug, gray=True)
        return img, bmask

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
                label_key = 'label-%09d' % index
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            # if len(label) > 2 and img.width*2 < img.height:
            #     print('vertical',label,img.width /img.height)
            #     return self[index + 1]

            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img.convert('RGB'))
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label, "bmask": bmask}
            return sample

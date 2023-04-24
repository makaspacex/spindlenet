# coding:utf-8
import random
import sys

import lmdb
import numpy as np
import six
from PIL import Image
from torch.utils.data import Dataset

from neko_2020nocr.dan.dataloaders.dataset_common import keepratio_resize


# ms : multi_sources
# semi: Semi supervision
class LmdbdatasetSingleLabeled(Dataset):
    def init_etc(self):
        pass

    def __init__(self, root, img_height=32, img_width=128,
                 transform=None, global_state='Test', maxT=25):
        self.maxT = maxT
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.global_state = global_state
        # Issue12
        self.target_ratio = img_width / float(img_height)
        env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.root = root
        self.env = env
        self.init_etc()
        pass

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if (index > len(self)):
            print('index range error')
            index = index % len(self) + 1
        if (index == 0):
            print("0 base detected. adding one")
            index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img = keepratio_resize(img, self.img_height, self.img_width, self.target_ratio, True)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
            sample = {'image': img, 'label': label}
            return sample


class LmdbdatasetSingleUnlabeled(Dataset):
    def init_etc(self):
        pass

    def __init__(self, root, img_height=32, img_width=128,
                 transform=None, global_state='Test', maxT=25):
        self.maxT = maxT
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.global_state = global_state
        # Issue12
        self.target_ratio = img_width / float(img_height)
        env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.root = root
        self.env = env
        self.init_etc()
        pass

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if (index > len(self)):
            print('index range error')
            index = index % len(self) + 1
        if (index == 0):
            print("0 base detected. adding one")
            index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img = keepratio_resize(img, self.img_height, self.img_width, self.target_ratio, True)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
            sample = {'image': img}
            return sample


class LmdbdatasetMs(Dataset):
    def init_etc(self):
        pass

    def init_call(self, datasets, ratio=None, global_state='Test', repeat=1):
        self.repos = []
        self.ratio = []
        self.global_state = global_state
        self.repeat = repeat
        self.nSamples = 0
        self.maxlen = 0
        for i in range(0, len(datasets)):
            self.repos.append(datasets[i])
            length = len(self.repos[-1])
            if (length > self.maxlen):
                self.maxlen = length
            self.nSamples += length
        if ratio != None:
            assert len(datasets) == len(datasets), 'length of ratio must equal to length of roots!'
            for i in range(0, len(datasets)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0, len(datasets)):
                self.ratio.append(self.repos[i].nSamples / float(self.nSamples))

    def __fromwhich__(self):
        rd = random.random()
        total = 0
        for i in range(0, len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(1, self.maxlen)
        index = index % len(self.repos[fromwhich])
        assert index <= len(self), 'index range error'
        index += 1
        sample = self.repos[fromwhich][index]
        return sample

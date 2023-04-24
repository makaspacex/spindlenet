# coding:utf-8
import numpy as np
import torch
from torch.utils.data import Dataset

from neko_2020nocr.dan.dataloaders.dataset_common import keepratio_resize
from neko_sdk.ocr_modules.trdg_driver.corpus_data_generator_driver import neko_random_string_generator
from neko_sdk.ocr_modules.trdg_driver.corpus_data_generator_driver import neko_skip_missing_string_generator


class Nekoolsdataset(Dataset):

    def load_random_generator(self, root, maxT):
        meta = torch.load(root)
        g = neko_random_string_generator(meta, meta["bgims"], max_len=maxT)
        self.nSamples = 19999999

        return g

    def __init__(self, root=None, ratio=None, img_height=32, img_width=128,
                 transform=None, global_state='Test', maxT=25):
        self.generator = self.load_random_generator(root)
        # Admit it=== you can never exhaust self. But let's at least set a number for an epoch
        self.global_state = global_state
        self.load_random_generator(root)
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        # Issue12
        self.target_ratio = img_width / float(img_height)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img, label = self.generator.random_clip()
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


class NekoOLSCDataset(Nekoolsdataset):
    def load_random_generator(self, root):
        meta = torch.load(root)
        g = neko_skip_missing_string_generator(meta, meta["bgims"])
        self.nSamples = g.nSamples
        return g

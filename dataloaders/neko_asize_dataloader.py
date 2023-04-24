import random

import cv2
import numpy as np
import six
from PIL import Image

from neko_2020nocr.dan.dataloaders.dataset_scene import colored_lmdbDataset
from neko_sdk.ocr_modules.augmentation.qhbaug import qhbwarp


# returns raw image,rectified_image and so on.
# let's make this a small 2-phasemodel
# we have a few anchors. The model selects on aspect ratios, and the model selects for sort edge size.

class colored_asize_lmdbDataset(colored_lmdbDataset):
    def keepratio_resize(self, img):
        cur_ratio = img.size[0] / float(img.size[1])
        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)

        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if (self.qhb_aug):
            try:
                img = qhbwarp(img, 10)
            except:
                pass
        dimg = cv2.resize(img, (mask_height, mask_width))
        return dimg, img

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

            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            # if len(label) > 2 and img.width*2 < img.height:
            #     print('vertical',label,img.width /img.height)
            #     return self[index + 1]
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                rim, im = self.keepratio_resize(img.convert('RGB'))
            except:

                print('Size error for %d' % index)
                return self[index + 1]
            if (len(rim.shape) == 2):
                rim = rim[:, :, np.newaxis]
                im = im[:, :, np.newaxis]
            if self.transform:
                rim = self.transform(rim)
            sample = {'rectimage': rim, "rawimage": im, 'label': label}
            return sample

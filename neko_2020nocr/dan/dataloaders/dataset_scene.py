# coding:utf-8
import random
import sys

import cv2
import lmdb
import numpy as np
import six
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from neko_sdk.ocr_modules.qhbaug import qhbwarp


class LmdbDataset(Dataset):
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

    def __fromwhich__(self):
        rd = random.random()
        total = 0
        for i in range(0, len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img):
        cur_ratio = img.size[0] / float(img.size[1])

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        if (self.qhb_aug):
            try:
                img = qhbwarp(img, 10)
            except:
                pass
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            # print("else",cur_ratio,self.target_ratio)
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img

        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
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
                img, bmask = self.keepratio_resize(img.convert('RGB'))
            except Exception as e:
                raise e
                print('Size error for %d' % index)
                return self[index + 1]
            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label, "bmask": bmask}
            return sample

class LmdbDataset_XJ(Dataset):
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
            raise Exception("this version dosent support ration settings")

        self.transform = transform
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

    def init_etc(self):
        pass

    def set_dss(self, roots):
        self.index_num_list = []
        self.txns = []
        for i, root in enumerate(roots):
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
            
            txn = env.begin(write=False)
            self.txns.append(txn)
        
        for i, txn in enumerate(self.txns):
            keys = list(txn.cursor().iternext(values=False))
            kk = [int(str(_.decode()).split("-")[-1])  for _ in keys if "num" not in str(_.decode())]
            num_indexs = sorted(list(set(kk)))
            num_indexs = [[i, k] for k in num_indexs]
            self.index_num_list += num_indexs
            
            nSamples = len(num_indexs)
            self.nSamples += nSamples
            
            self.envs.append(env)
            self.init_etc()

    def keepratio_resize(self, img):
        cur_ratio = img.size[0] / float(img.size[1])

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        if (self.qhb_aug):
            try:
                img = qhbwarp(img, 10)
            except:
                pass
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            # print("else",cur_ratio,self.target_ratio)
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img

        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
        return img, bmask

    def __len__(self):
        return len(self.index_num_list)

    def __getitem__(self, item):
        txn_i, index = self.index_num_list[item]
        txn = self.txns[txn_i]
        
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
            img, bmask = self.keepratio_resize(img.convert('RGB'))
        except Exception as e:
            raise e
            print('Size error for %d' % index)
            return self[index + 1]
        if (len(img.shape) == 2):
            img = img[:, :, np.newaxis]
        if self.transform:
            img = self.transform(img)
            bmask = self.transform(bmask)

        sample = {'image': img, 'label': label, "bmask": bmask}
        return sample


class ColoredLmdbDatasetT(LmdbDataset):
    def clahe(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return bgr

    def keepratio_resize(self, img):

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        # img=self.clahe(img)
        #
        # if(img.shape[0]>img.shape[1]*2):
        #     img=np.transpose(img,[1,0,2])
        cur_ratio = img.shape[1] / float(img.shape[0])

        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if (self.qhb_aug):
            try:
                img = qhbwarp(img, 10)
            except:
                pass

        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            # print("else",cur_ratio,self.target_ratio)
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        img = mask
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        return img, bmask

class ColoredLmdbDatasetTV(LmdbDataset_XJ):
    def clahe(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return bgr

    def keepratio_resize(self, img):

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        # img=self.clahe(img)
        #
        # if(img.shape[0]>img.shape[1]*2):
        #     img=np.transpose(img,[1,0,2])
        cur_ratio = img.shape[1] / float(img.shape[0])

        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if (self.qhb_aug):
            try:
                img = qhbwarp(img, 10)
            except:
                pass

        cur_target_height,cur_target_width=resize_v_align(
            cur_ratio,self.target_ratio,self.img_height,self.img_width
        )
 

            # print("else",cur_ratio,self.target_ratio)
            
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        img = mask
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        return img, bmask


class ColoredLmdbDataset(LmdbDataset):
    #TODO 针对横向和
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

        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            # print("else",cur_ratio,self.target_ratio)
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
        return img, bmask

def resize_v_align(cur_ratio,target_ratio,img_height,img_width):
    if cur_ratio < target_ratio:
        cur_target_height=img_height;
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = int(img_height * cur_ratio);
    else:
        cur_target_width = img_width
        cur_target_height = int(img_width/cur_ratio);
    return cur_target_height,cur_target_width;
 
class ColoredLmdbDatasetV(LmdbDataset_XJ):
    #TODO 针对横向和
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
        cur_target_height,cur_target_width=resize_v_align(
            cur_ratio,self.target_ratio,self.img_height,self.img_width
        )
        
            # print("else",cur_ratio,self.target_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
        return img, bmask


class ColoredLmdbdatasetSemi(ColoredLmdbDataset):
    def __init__(self, roots=None, cased_annoatations=None, ratio=None, img_height=32, img_width=128,
                 transform=None, global_state='Test', maxT=25, repeat=1, qhb_aug=False):
        super().__init__(roots, ratio, img_height, img_width, transform, global_state, maxT, repeat, qhb_aug)
        self.cased = cased_annoatations

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
                img, bmask = self.keepratio_resize(img.convert('RGB'))
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label, "cased": self.cased[fromwhich]}
            return sample


class LmdbDatasetSemi(LmdbDataset):
    def __init__(self, roots=None, cased_annoatations=None, ratio=None, img_height=32, img_width=128,
                 transform=None, global_state='Test', maxT=25, repeat=1, qhb_aug=False):
        super().__init__(roots, ratio, img_height, img_width, transform, global_state, maxT,
                                              repeat, qhb_aug)
        self.cased = cased_annoatations

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
                img, bmask = self.keepratio_resize(img.convert('RGB'))
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label, "cased": self.cased[fromwhich]}
            return sample


# some chs datasets are too smol.
class LmdbDatasetRepeat(LmdbDataset):
    def __len__(self):
        return self.nSamples * self.repeat

    def __getitem__(self, index):
        index %= self.nSamples
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
                print("reponame: ", self.roots[fromwhich])
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print("reponame: ", self.roots[fromwhich])
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)
            sample = {'image': img, 'label': label, "bmask": bmask}
            return sample


class LmdbDatasetRepeatH(LmdbDataset):
    def __len__(self):
        return self.nSamples * self.repeat

    def __getitem__(self, index):
        index %= self.nSamples
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
                print("reponame: ", self.roots[fromwhich])
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print("reponame: ", self.roots[fromwhich])
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            if (self.novert and img.size[0] / float(img.size[1]) < 1):
                # print(img.size)
                # print("vertical image")
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label, "bmask": bmask}
            return sample


class LmdbDatasetRepeatS(LmdbDataset):
    def __len__(self):
        if (self.repeat == -1):
            return 2147483647
        else:
            return self.nSamples * self.repeat

    def init_etc(self):
        self.ccache = {}
        self.occr_cnt = {}

    def grab(self, fromwhich, index):
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
                print("reponame: ", self.roots[fromwhich])
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print("reponame: ", self.roots[fromwhich])
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img, mask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                mask = self.transform(mask)
            sample = {'image': img, 'label': label, "bmask": mask}

            return sample

    def __getitem__(self, index):
        index %= self.nSamples
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        # prevent the common characters smacking the space.
        if len(self.ccache) and random.randint(0, 10) > 7:
            ks = list(self.ccache.keys())
            fs = [1. / self.occr_cnt[k] for k in ks]
            k = random.choices(ks, fs)[0]
            # print(k,self.occr_cnt[k])
            ridx, fwhich = self.ccache[k]
            sample = self.grab(ridx, fwhich)
        else:
            sample = self.grab(fromwhich, index)
            minoccr = np.inf
            minK = None
            for l in sample["label"]:
                if l not in self.occr_cnt:
                    self.occr_cnt[l] = 0
                self.occr_cnt[l] += 1
                if (self.occr_cnt[l] < minoccr):
                    minoccr = self.occr_cnt[l]
                    minK = l
            if (minK):
                self.ccache[minK] = (fromwhich, index)

        for l in sample["label"]:
            self.occr_cnt[l] += 1
        return sample


class ColoredLmdbDatasetRepeatS(LmdbDatasetRepeatS):
    def grab(self, fromwhich, index):
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
                print("reponame: ", self.roots[fromwhich])
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print("reponame: ", self.roots[fromwhich])
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            try:
                img, bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            # img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label, "bmask": bmask}

            return sample

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

        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            # print("else",cur_ratio,self.target_ratio)
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
        return img, bmask


class LmdbDatasetRepeatHS(LmdbDatasetRepeatS):
    def __len__(self):
        if (self.repeat < 0):
            return 2147483647
        return self.nSamples * self.repeat

    def init_etc(self):
        self.ccache = {}
        self.occr_cnt = {}

    def grab(self, fromwhich, index):
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
                print("reponame: ", self.roots[fromwhich])
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print("reponame: ", self.roots[fromwhich])
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            if (self.novert and img.size[0] / float(img.size[1]) < 1):
                # print(img.size)
                # print("vertical image")
                return self[index + 1]
            try:
                img, mask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]

            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                mask = self.transform(mask)
            sample = {'image': img, 'label': label, "bmask": mask}

            return sample


class ColoredLmdbDatasetRepeatHS(LmdbDatasetRepeatHS):
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

        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            # print("else",cur_ratio,self.target_ratio)
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
        return img, bmask

    def grab(self, fromwhich, index):
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
                print("reponame: ", self.roots[fromwhich])
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'))
            except:
                print("reponame: ", self.roots[fromwhich])
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            if (self.novert and img.size[0] / float(img.size[1]) < 1):
                # print(img.size)
                # print("vertical image")
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            # img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)
            sample = {'image': img, 'label': label, "bmask": bmask}

            return sample


class Lmdbdatasettransform(LmdbDataset):
    def __init__(self, roots=None, ratio=None, img_height=32, img_width=128, transform=None, global_state='Test',
                 maxT=25, repeat=1):
        super().__init__(roots, ratio, img_height, img_width, transform, global_state, maxT, repeat)
        self.envs = []
        self.roots = []
        self.maxT = maxT
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.global_state = global_state
        self.repeat = repeat
        self.totensor = torchvision.transforms.ToTensor()
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
        self.target_ratio = img_width / float(img_height)

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
                img = self.transform(img)
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]
            if self.transform:
                img = self.totensor(img)
                bmask = self.totensor(bmask)
            sample = {'image': img, 'label': label, "bmask": bmask}
            return sample

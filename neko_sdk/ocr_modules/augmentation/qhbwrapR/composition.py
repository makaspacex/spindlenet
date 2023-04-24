import random

import cv2
import numpy as np

from neko_sdk.ocr_modules.augmentation.qhbwrapR.execution import neko_qhbwrapR_executer


# While this implements better fairness, but NOTHING is guaranteed...
# Different version of libraries can have different results for the same set of parameters.
# We cannot afford record all augmented images, as they would be too big.

def random_tia_distort(img_h, img_w, segment=4):
    cut = img_w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append(
        [np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            np.random.randint(thresh) - half_thresh
        ])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            img_h + np.random.randint(thresh) - half_thresh
        ])

    dst_pts = np.clip(np.array(dst_pts), [0, 0], [img_w, img_h])

    return src_pts, dst_pts, img_w, img_h


def random_tia_stretch(img_h, img_w, segment=4):
    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    return src_pts, dst_pts, img_w, img_h


def random_tia_perspective(img_h, img_w):
    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    return src_pts, dst_pts, img_w, img_h


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def random_jitter_param(h, w):
    s = -1
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
    return [s]


def random_gaussian_noise(image, mean=0, var=0.1):
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    return [noise]


def random_cvt_color_param():
    delta = 0.001 * random.random() * flag()
    return [delta]


def random_crop_para(h, w):
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    ratio = random.randint(0, 1)
    return top_crop, ratio


class neko_qhbwarp_composer:
    """
    Config
    """

    def addact(self, img, action, acts):
        img = self.executer.execute_cfg(img, action)
        h, w, _ = img.shape
        acts.append(action)
        return acts, img, h, w

    # We have to tag along to get the real image size...
    # What we do not do is
    def random_config(self, img, prob):
        h, w, _ = img.shape
        acts = []
        if self.distort:  # 扭曲
            if self.rng.random() <= prob and h >= 20 and w >= 20:
                a = {"act": "tia",
                     "para": random_tia_distort(h, w, self.rng.randint(3, 6))}
                acts, img, h, w = self.addact(img, a, acts)

        if self.stretch:  # 拉伸
            if self.rng.random() <= prob and h >= 20 and w >= 20:
                a = {"act": "tia",
                     "para": random_tia_stretch(h, w, self.rng.randint(3, 6))}
                acts, img, h, w = self.addact(img, a, acts)
        if self.perspective:  # 透视变换
            if self.rng.random() <= prob:
                a = {"act": "tia",
                     "para": random_tia_perspective(h, w)}
                acts, img, h, w = self.addact(img, a, acts)

        if self.crop:  # 随机裁剪
            if self.rng.random() <= prob and h >= 20 and w >= 20:
                a = {"act": "tia",
                     "para": random_crop_para(h, w)}
                acts, img, h, w = self.addact(img, a, acts)

        if self.blur:  # 高斯模糊
            if self.rng.random() <= prob:
                a = {"act": "blur",
                     "para": []}
                acts, img, h, w = self.addact(img, a, acts)

        if self.cvtcolor:
            if self.rng.random() <= prob:
                a = {"act": "cvtcolor",
                     "para": random_cvt_color_param()}
                acts, img, h, w = self.addact(img, a, acts)
        if self.jitter:
            if self.rng.random() <= prob:
                a = {"act": "jitter",
                     "para": random_jitter_param(h, w)
                     }
                acts, img, h, w = self.addact(img, a, acts)
        # if self.noise:  # 添加高斯噪声
        #     if self.rng.random() <= prob:
        #         a = {"act": "gaussian_noise",
        #              "para": random_gaussian_noise(img)
        #              }
        #         acts, img, h, w = self.addact(img, a, acts)

        if self.reverse:  # 颜色翻转
            if self.rng.random() <= prob:
                a = {"act": "reverse",
                     "para": []
                     }
                acts, img, h, w = self.addact(img, a, acts)

        return acts, img

    def __init__(self, use_tia):
        self.distort = use_tia
        self.stretch = use_tia
        self.perspective = use_tia
        self.use_tia = use_tia

        self.executer = neko_qhbwrapR_executer()

        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.crop = False
        self.affine = True
        self.reverse = False
        self.noise = True
        self.jitter = True
        self.blur = True
        self.cvtcolor = True
        self.rng = random


if __name__ == '__main__':
    image = cv2.imread('/run/media/lasercat/risky/sample.png')
    print(image.shape)
    composer = neko_qhbwarp_composer(True)
    aug, image = composer.random_config(image, 0.4)
    print(image.shape)
    print(aug)
    cv2.imwrite('/run/media/lasercat/risky/samplea.png', image)

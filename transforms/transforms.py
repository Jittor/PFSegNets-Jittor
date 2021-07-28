import random
import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import jittor as jt
from PIL import Image
from config import cfg
from scipy.ndimage.interpolation import shift

from skimage.segmentation import find_boundaries


class MaskToTensor(object):
    def __call__(self, img):
        return jt.Var(np.array(img, dtype=np.int32)).long()


class RelaxedBoundaryLossToTensor(object):
    """
    Boundary Relaxation
    """

    def __init__(self, ignore_id, num_classes):
        self.ignore_id = ignore_id
        self.num_classes = num_classes

    def new_one_hot_converter(self, a):
        ncols = self.num_classes+1
        out = np.zeros((a.size, ncols), dtype=np.uint8)
        out[np.arange(a.size), a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out

    def __call__(self, img):

        img_arr = np.array(img)
        img_arr[img_arr == self.ignore_id] = self.num_classes

        if cfg.STRICTBORDERCLASS != None:
            one_hot_orig = self.new_one_hot_converter(img_arr)
            mask = np.zeros((img_arr.shape[0], img_arr.shape[1]))
            for cls in cfg.STRICTBORDERCLASS:
                mask = np.logical_or(mask, (img_arr == cls))
        one_hot = 0

        border = cfg.BORDER_WINDOW
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            border = border // 2
            border_prediction = find_boundaries(
                img_arr, mode='thick').astype(np.uint8)

        for i in range(-border, border+1):
            for j in range(-border, border+1):
                shifted = shift(img_arr, (i, j), cval=self.num_classes)
                one_hot += self.new_one_hot_converter(shifted)

        one_hot[one_hot > 1] = 1

        if cfg.STRICTBORDERCLASS != None:
            one_hot = np.where(np.expand_dims(mask, 2), one_hot_orig, one_hot)

        one_hot = np.moveaxis(one_hot, -1, 0)

        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            one_hot = np.where(border_prediction, 2*one_hot, 1*one_hot)
            # print(one_hot.shape)
        return jt.Var(one_hot).byte()


class RandomGaussianBlur(object):
    """
    Apply Gaussian Blur
    """

    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


class RandomBilateralBlur(object):
    """
    Apply Bilateral Filtering

    """

    def __call__(self, img):
        sigma = random.uniform(0.05, 0.75)
        blurred_img = denoise_bilateral(
            np.array(img), sigma_spatial=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))

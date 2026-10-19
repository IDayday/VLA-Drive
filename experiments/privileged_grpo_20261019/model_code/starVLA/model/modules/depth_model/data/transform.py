import os
import cv2
import numpy as np
from ppd.utils.logger import Log
import torch
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig
EPS = 1e-4

class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __str__(self):
        return "PrepareForNet"

    def __repr__(self):
        return "PrepareForNet"

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.uint8)
            sample["mask"] = np.ascontiguousarray(sample["mask"])[None]

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)[None]

        return sample


def cv2_resize(image, size, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(image, size, interpolation=interpolation)[None]



class Resize(object):
    """Resize sample to given size (width, height).
    """
    def __init__(
        self,
        width=None,
        height=None,
        # image_interpolation_method=cv2.INTER_AREA,
        image_interpolation_method = cv2.INTER_LINEAR,
    ):
        self.width = width
        self.height = height
        self.__image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        width, height = self.width, self.height
        if width == sample['image'].shape[1] and height == sample['image'].shape[0]:
            return sample
        Log.debug(
            'Resize: {} -> {}'.format(sample["image"].shape, (height, width)))
        # resize sample
        ori_height, ori_width = sample['image'].shape[:2]
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )
        if "depth" in sample:
            sample["depth"] = cv2.resize(
                sample["depth"], 
                (width, height), 
                interpolation=cv2.INTER_NEAREST)

        if "mask" in sample:
            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32), 
                (width, height), 
                interpolation=cv2.INTER_NEAREST)
        return sample

class Resize_4K_Crop(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width=None,
        height=None,
        crop_type='random',
        image_interpolation_method=cv2.INTER_AREA,
    ):
        self.width = width
        self.height = height
        self.crop_type = crop_type
        self.__image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        width, height = 1920, 1024
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        # crop sample
        crop_h = self.height
        crop_w = self.width
        if self.crop_type == 'random':
            # random crop
            top = np.random.randint(0, height - crop_h + 1)
            left = np.random.randint(0, width - crop_w + 1)
        else:
            # center crop
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
        sample["image"] = sample["image"][top:top+crop_h, left:left+crop_w]

        if "depth" in sample:
            sample["depth"] = cv2.resize(
                sample["depth"], (width, height), 
                interpolation=cv2.INTER_NEAREST
            )
            # crop sample
            sample["depth"] = sample["depth"][top:top+crop_h, left:left+crop_w]
        if "mask" in sample:
            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"][top:top+crop_h, left:left+crop_w]
        return sample
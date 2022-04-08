from logging import exception, raiseExceptions
from os import path
from tkinter.messagebox import NO
import torch 
import numpy as np
from random import randint
import cv2
from torch._C import clear_autocast_cache
from utils.utils_blindsr import degradation_bsrgan, degradation_bsrgan_plus_an,random_crop
from torchvision import transforms, utils
import re
import collections
from torch._six import string_classes
import random
from utils import utils_image,utils_blindsr
np_str_obj_array_pattern = re.compile(r'[SaUO]')


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):


        img_origin = sample["img_origin"]
        img_noisy = sample["img_noisy"]

        image_o = np.float32(img_origin/255.)
        image_n = np.float32(img_noisy/255.)

        sample_ = {"img_H": image_o,"img_L":image_n}
  
        return sample_


class Rotate(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,degree = None):
        self.degree = degree
        


    def __call__(self, sample):

        if(self.degree == None):
            return sample

        
        image_o = cv2.rotate(sample["img_H"],  self.degree)
        image_n = cv2.rotate(sample["img_L"],  self.degree)

        sample_ = {"img_H": image_o,"img_L":image_n}
  
        return sample_

import os

class Degradate(object):
    def __init__(self,scale = 4,patch_size_w = 64,patch_size_h=64,use_same = False,use_inter = False,use_whole_image = False):
        self.scale = scale
        self.patch_size_w = patch_size_w
        self.patch_size_h= patch_size_h
        self.use_whole_image = use_whole_image

    def __call__(self, sample):
        img_origin = sample["img_H"]
        img_noisy = sample["img_L"]
        
        img_L=None
        img_H=None
        h1, w1 = img_origin.shape[:2]
        interpolate= random.choice([1, 2, 3])
        height = self.patch_size_h * self.scale
        width = self.patch_size_w * self.scale

        if(self.use_whole_image):
            h = height
            w = width
            if(self.patch_size_h> self.patch_size_w):
                while(h < height and w < width):
                    h = h +50
                    w = h/h1 * w1
            else:
                while(h < height and w < width):
                    w = w +50
                    h = w/w1 * h1  

            img_L = cv2.resize(img_noisy, (round(w/self.scale), round(h/self.scale)), interpolation=interpolate)
            img_H = cv2.resize(img_origin, (round(w), round(h)), interpolation=interpolate)

            img_L,img_H = utils_blindsr.random_crop(img_L,img_H,sf=self.scale,lq_patchsize_w=self.patch_size_w,lq_patchsize_h=self.patch_size_h)        
        else:
            if(h1 < height or w1 < width):
                h = h1
                w= w1
                while(h < height or w < width):
                    h = h +50
                    w = h/h1 * w1

                img_origin = cv2.resize(img_origin, (round(w), round(h)), interpolation=cv2.INTER_CUBIC)
                img_noisy = cv2.resize(img_noisy, (round(w), round(h)), interpolation=cv2.INTER_CUBIC)

            img_L, img_H = degradation_bsrgan_plus_an(img=img_noisy,hq=img_origin, sf=self.scale, lq_patchsize_w=self.patch_size_w,lq_patchsize_h=self.patch_size_h,degrade=True,noise=False)


        # cv image: H x W x C 
        return {'img_L': img_L,
                'img_H': img_H}




class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self._size = output_size
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imagehr = sample['img_H']
        if(imagehr.shape[0] < self._size or imagehr.shape[1] < self._size):
            raise Exception("image dimenstion is low") 

        print("image shape",imagehr.shape)
        h, w = imagehr.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = imagehr[top: top + new_h,
                      left: left + new_w]

 

        return {'img_H': imagehr}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imagelr, imagehr = sample['img_L'], sample['img_H']

        #convert to tensor
        img_L_ = torch.from_numpy(imagelr)
        img_H_ = torch.from_numpy(imagehr)
        # torch image: C x H x W
        img_L = img_L_.permute(2, 0, 1).float()
        img_H = img_H_.permute(2, 0, 1).float()

        return {'img_L': img_L,
                'img_H': img_H}








default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")




class DataBatch:
    def __init__(self,transfrom,scale,max_box,max_cells,devider=2,force_size = None,use_whole_image=False):
        self.transfrom = transfrom
        self.scale = scale
        self.max_box = max_box
        self.max_cells = max_cells
        self.rotate_degree = [None,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_CLOCKWISE]
        self.devider = devider
        self.force_size = force_size
        self.use_whole_image =use_whole_image


    def collate_fn(self,batch):
        
        #calculate img crop size
        min_h,max_h,min_w,max_w = self.max_box
        patch_h = 2000
        patch_w = 2000
        while(patch_h*patch_w > self.max_cells or (patch_h%self.devider !=0 or patch_w%self.devider !=0)):
            patch_h = randint(min_h,max_h)
            patch_w = randint(min_w,max_w)
        if(self.force_size is not None):
            patch_h = self.force_size
            patch_w = self.force_size
        degrade = Degradate(scale=self.scale,patch_size_w=patch_w,patch_size_h=patch_h,use_whole_image=self.use_whole_image)
        rotate = Rotate(degree=self.rotate_degree[randint(0,3)])
        batch_= []
        for sample in batch:
            sample_ = degrade(sample)
            sample_ = rotate(sample_)
            # utils_image.imsave(sample_["img_H"]*255, os.path.join('testsets/exported', 'img_H'+'.png'))
            # utils_image.imsave(sample_["img_L"]*255, os.path.join('testsets/exported', 'img_L'+'.png'))

            sample_ = self.transfrom(sample_)
            batch_.append(sample_)
        
        # elem = batch[0]
        # return {key: self.cellect([d[key] for d in batch]) for key in elem}
        return self.default_collate(batch_)

    # def cellect(self,batch):
    #     elem = batch[0]
    #     elem_type = type(elem)
    #     out = None
    #     if torch.utils.data.get_worker_info() is not None:
    #         # If we're in a background process, concatenate directly into a
    #         # shared memory tensor to avoid an extra copy
    #         numel = sum(x.numel() for x in batch)
    #         storage = elem.storage()._new_shared(numel)
    #         out = elem.new(storage)
    #     return torch.stack(batch, 0, out=out)


    def default_collate(self,batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            try:
                return elem_type({key: self.default_collate([d[key] for d in batch]) for key in elem})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self.default_collate(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([self.default_collate(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self.default_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))


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


        img_H = sample["img_H"]

        img_H = np.float32(img_H/255.)

        sample_ = {"img_H": img_H}
  
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

class Crop(object):
    def __init__(self,patch_size_w = 64,patch_size_h=64):

        self.patch_size_w = patch_size_w
        self.patch_size_h= patch_size_h


    def __call__(self, sample):
        img_origin = sample["img_H"]
        
        img_L=None
        img_H=img_origin.copy()

      
        h1, w1 = img_H.shape[:2]
        interpolate= random.choice([1, 2, 3])
        h ,w = None,None

        # if(h1 >= w1 and self.patch_size_h >= self.patch_size_w and w1 >= self.patch_size_w and h1 >= self.patch_size_h):
        #     w = self.patch_size_h
        #     h = w/w1 * h1  
        # elif(h1 >= w1 and self.patch_size_h >= self.patch_size_w and w1 >= self.patch_size_w and h1 <= self.patch_size_h):
        #     h = self.patch_size_h
        #     w = h/h1 * w1
        # elif(h1 >= w1 and self.patch_size_h >= self.patch_size_w and w1 <= self.patch_size_w and h1 >= self.patch_size_h):
        #     w = self.patch_size_h
        #     h = w/w1 * h1
        # elif(h1 >= w1 and self.patch_size_h >= self.patch_size_w and w1 <= self.patch_size_w and h1 <= self.patch_size_h):
        #     w = self.patch_size_h
        #     h = w/w1 * h1
        # elif(h1 >= w1 and self.patch_size_h <= self.patch_size_w and w1 >= self.patch_size_w and h1 >= self.patch_size_h):
        #     w = self.patch_size_w
        #     h = w/w1 * h1
        # elif(h1 >= w1 and self.patch_size_h <= self.patch_size_w and w1 >= self.patch_size_w and h1 <= self.patch_size_h):
        #     h = self.patch_size_h
        #     w = h/h1 * w1
        # elif(h1 >= w1 and self.patch_size_h <= self.patch_size_w and w1 <= self.patch_size_w and h1 >= self.patch_size_h):
        #     w = self.patch_size_w
        #     h = w/w1 * h1   
        # elif(h1 >= w1 and self.patch_size_h <= self.patch_size_w and w1 <= self.patch_size_w and h1 <= self.patch_size_h):
        #     w = self.patch_size_w
        #     h = w/w1 * h1  

        # elif(h1 < w1 and self.patch_size_h >= self.patch_size_w and w1 >= self.patch_size_w and h1 >= self.patch_size_h):
        #     h = self.patch_size_h
        #     w = h/h1 * w1
        # elif(h1 < w1 and self.patch_size_h >= self.patch_size_w and w1 >= self.patch_size_w and h1 <= self.patch_size_h):
        #     h = self.patch_size_h
        #     w = h/h1 * w1
        # elif(h1 < w1 and self.patch_size_h >= self.patch_size_w and w1 <= self.patch_size_w and h1 >= self.patch_size_h):
        #     w = self.patch_size_w
        #     h = w/w1 * h1
        # elif(h1 < w1 and self.patch_size_h >= self.patch_size_w and w1 <= self.patch_size_w and h1 <= self.patch_size_h):
        #     h = self.patch_size_h
        #     w = h/h1 * w1
        # elif(h1 < w1 and self.patch_size_h <= self.patch_size_w and w1 >= self.patch_size_w and h1 >= self.patch_size_h):
        #     h = self.patch_size_w
        #     w = h/h1 * w1  
        # elif(h1 < w1 and self.patch_size_h <= self.patch_size_w and w1 >= self.patch_size_w and h1 <= self.patch_size_h):
        #     h = self.patch_size_h
        #     w = h/h1 * w1
        # elif(h1 < w1 and self.patch_size_h <= self.patch_size_w and w1 <= self.patch_size_w and h1 >= self.patch_size_h):
        #     w = self.patch_size_w
        #     h = w/w1 * h1   
        # elif(h1 < w1 and self.patch_size_h <= self.patch_size_w and w1 <= self.patch_size_w and h1 <= self.patch_size_h):
        #     w = self.patch_size_w
        #     h = w/w1 * h1        
        # else:
        #     print("odd")
        
        w = self.patch_size_w
        h = self.patch_size_h
        if(self.patch_size_h> self.patch_size_w):
            while(h < self.patch_size_h and w < self.patch_size_w):
                h = h +50
                w = h/h1 * w1
        else:
            while(h < self.patch_size_h and w < self.patch_size_w):
                w = w +50
                h = w/w1 * h1  
            



        # print("image",h1,w1)
        # print("patch",self.patch_size_h,self.patch_size_w)
        # print("dim_caluculated",h,w)
        img_H = cv2.resize(img_H, (round(w), round(h)), interpolation=interpolate)

        img_L,img_H = utils_blindsr.random_crop(img_H,img_H,sf=1,lq_patchsize_w=self.patch_size_w,lq_patchsize_h=self.patch_size_h)
        # print("img_L",img_L.shape)
        # print("img_H",img_H.shape)



        # cv image: H x W x C 
        return {'img_L': img_L,
                'img_H': img_H}

class CvtToGRAY(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img_L ,img_H= sample["img_L"], sample['img_H']
        img_L = cv2.cvtColor(img_L,cv2.COLOR_BGR2GRAY)
        img_L = cv2.cvtColor(img_L,cv2.COLOR_GRAY2BGR)
  
        return {'img_H': img_H,'img_L':img_L}



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
    def __init__(self,transfrom,scale,max_box,max_cells,devider=2,force_size = None):
        self.transfrom = transfrom
        self.scale = scale
        self.max_box = max_box
        self.max_cells = max_cells
        self.rotate_degree = [None,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_CLOCKWISE]
        self.devider = devider
        self.force_size = force_size


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

        cvt2gray = CvtToGRAY()    
        crop = Crop(patch_size_w=patch_w,patch_size_h=patch_h)
        rotate = Rotate(degree=self.rotate_degree[randint(0,3)])

        batch_= []
        for sample in batch:
            sample_ = crop(sample)
            sample_ = cvt2gray(sample_)
            sample_ = rotate(sample_)
            #utils_image.imsave(sample_["img_H"]*255, os.path.join('testsets/exported', 'img_H'+'.png'))
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


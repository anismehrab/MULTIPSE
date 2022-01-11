from logging import exception, raiseExceptions
from os import path
import torch 
import numpy as np
from random import randint
import cv2
from torch._C import clear_autocast_cache
from utils.utils_blindsr import degradation_bsrgan, degradation_bsrgan_plus_an
from torchvision import transforms, utils
import re
import collections
from torch._six import string_classes
import random
np_str_obj_array_pattern = re.compile(r'[SaUO]')


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class DataBatch:
    def __init__(self,transfrom,max_box,max_cells,scale=4):
        self.transfrom = transfrom
        self.max_box = max_box
        self.max_cells = max_cells
        self.scale = scale
        


    def collate_fn(self,batch):
        
        #calculate img crop size
        min_h,max_h,min_w,max_w = self.max_box
        patch_h = 2000
        patch_w = 2000
        while(patch_h*patch_w > self.max_cells or (patch_h%self.scale !=0 or patch_w%self.scale !=0)):
            patch_h = randint(min_h,max_h) 
            patch_w = patch_h #randint(min_w,max_w)

        rescale = AnimeRescale((patch_h,patch_w))

        batch_= []
        for sample in batch:
            sample_ = rescale(sample)
            sample_ = self.transfrom(sample_)
            # print(sample_["img_H"].size())
            # print(sample_["img_L"].size())

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





class AnimeRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self ,img_size):
        assert isinstance(img_size, (int, tuple))
        self.input_size = img_size
  

    def __call__(self, sample):

        img_A,img_O = sample['img_A'] , sample["img_O"]

        h, w = img_O.shape[:2]
        if isinstance(self.input_size, int):
            if h > w:
                new_h, new_w = self.input_size * h / w, self.input_size
            else:
                new_h, new_w = self.input_size, self.input_size * w / h
        else:
            new_h, new_w = self.input_size

        new_h, new_w = int(new_h), int(new_w)
        
        img_O = cv2.resize(img_O, (new_h, new_w), interpolation=cv2.INTER_CUBIC)

        img_A = cv2.resize(img_A, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        # print("img_H_",img_H_.shape)

        return {'img_A': img_A, 'img_O': img_O}     


class AnimeNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img_A,img_O = sample['img_A'] , sample["img_O"]

        img_A = np.float32(img_A/255.)
        img_O = np.float32(img_O/255.)

        sample_ ={'img_A': img_A, 'img_O': img_O}                 
  
        return sample_


class AnimeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_A,img_O = sample['img_A'] , sample["img_O"]

        #convert to tensor
        img_O = torch.from_numpy(img_O)
        img_A = torch.from_numpy(img_A)
        # torch image: C x H x W
        img_O = img_O.permute(2, 0, 1).float()
        img_A = img_A.permute(2, 0, 1).float()
        # print("img_L",img_L.shape)
        # print("img_H",img_H.shape)

        return {'img_H': img_A, 'img_L': img_O}         
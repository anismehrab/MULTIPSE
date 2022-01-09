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
        self.rotate_degree = [None,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_CLOCKWISE]
        


    def collate_fn(self,batch):
        
        #calculate img crop size
        min_h,max_h,min_w,max_w = self.max_box
        patch_h = 2000
        patch_w = 2000
        while(patch_h*patch_w > self.max_cells or (patch_h%self.scale !=0 or patch_w%self.scale !=0)):
            patch_h = randint(min_h,max_h) 
            patch_w = randint(min_w,max_w)
        rescale = FaceRescale((patch_h,patch_w),(patch_h,patch_w))

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





class FaceRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self ,input_size,output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.input_size = input_size
  

    def __call__(self, sample):

        img_H= sample['img_H']

        h, w = img_H.shape[:2]
        if(h == self.input_size and w == self.input_size):
            return {'img_H': img_H, 'img_L': img_H}  

        if isinstance(self.input_size, int):
            if h > w:
                new_h, new_w = self.input_size * h / w, self.input_size
            else:
                new_h, new_w = self.input_size, self.input_size * w / h
        else:
            new_h, new_w = self.input_size

        new_h, new_w = int(new_h), int(new_w)
        
        img_L_ = cv2.resize(img_H, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        # print("img_L",img_L_.shape)

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img_H_ = cv2.resize(img_H, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        # print("img_H_",img_H_.shape)



    

        return {'img_H': img_H_, 'img_L': img_L_}     










class AddMaskFace(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.output_size = None
        self.masks = [self.line,self.rectangle,self.circle]

    def __call__(self, sample):
        img_H,img_L = sample['img_H'] , sample["img_L"]

        img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
        h, w = img_L.shape[:2]
        self.output_size = min(h,w)
        img_L = self.masks[0](img_L)
        for i in range(randint(2,4)):
            img_L = self.masks[2](img_L)

        # print("img_L",img_L.shape)

        return {'img_H': img_H,'img_L': img_L}

    def line(self,image):
        offset = self.output_size / 9
        s_h = randint(offset,self.output_size-offset)
        s_w = randint(offset,self.output_size-offset)
        e_h = randint(s_h,self.output_size-offset)
        e_w = randint(s_w,self.output_size-offset)

        img_masked = cv2.line(
            image,
            pt1 = (s_w, s_h), pt2 = (e_w, e_h),
            color = (255, 255, 255),
            thickness = randint(5,20))
        return img_masked    
    
    def rectangle(self,image):
        s_h = randint(0,int(self.output_size/3)-10)
        s_w = randint(0,int(self.output_size/3)-10)
        e_h = randint(s_h,int(self.output_size/3))
        e_w = randint(s_w,int(self.output_size/3))
        
        img_masked = cv2.rectangle(
                image,
                pt1 = (s_w, s_h), pt2 = (e_w, e_h),
                color = (255, 255, 255),
                thickness = -1)

        return img_masked 

    def circle(self,image):
        s_h = randint(self.output_size/2-(self.output_size/3),self.output_size-10)
        s_w = randint(self.output_size/2-(self.output_size/3),self.output_size-10)
        raduis = randint(5,int(min(self.output_size - max(s_h,s_w),self.output_size/20)))
        
        img_masked = cv2.circle(
                    image,
                    center = (s_w, s_h),
                    radius = raduis,
                    color = (255, 255, 255),
                    thickness = -1
                    )

        return img_masked


class FaceNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img_H,img_L = sample['img_H'] , sample["img_L"]

        img_H = np.float32(img_H/255.)
        img_L = np.float32(img_L/255.)

        sample_ ={'img_H': img_H, 'img_L': img_L}               
  
        return sample_


class FaceToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_H,img_L = sample['img_H'] , sample["img_L"]

        #convert to tensor
        img_H = torch.from_numpy(img_H)
        img_L = torch.from_numpy(img_L)
        # torch image: C x H x W
        img_L = img_L.permute(2, 0, 1).float()
        img_H = img_H.permute(2, 0, 1).float()
        # print("img_L",img_L.shape)
        # print("img_H",img_H.shape)

        return {'img_H': img_H,'img_L': img_L}        
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



class Degradate(object):
    def __init__(self,scale = 4,patch_size_w = 64,patch_size_h=64):
        self.scale = scale
        self.patch_size_w = patch_size_w
        self.patch_size_h= patch_size_h

    def __call__(self, sample):
        img_origin = sample["img_H"]
        img_noisy = sample["img_L"]

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
    def __init__(self,transfrom,scale,max_box,max_cells,devider=2):
        self.transfrom = transfrom
        self.scale = scale
        self.max_box = max_box
        self.max_cells = max_cells
        self.rotate_degree = [None,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_CLOCKWISE]
        self.devider = devider


    def collate_fn(self,batch):
        
        #calculate img crop size
        min_h,max_h,min_w,max_w = self.max_box
        patch_h = 2000
        patch_w = 2000
        while(patch_h*patch_w > self.max_cells or (patch_h%self.devider !=0 or patch_w%self.devider !=0)):
            patch_h = randint(min_h,max_h)
            patch_w = randint(min_w,max_w)

        degrade = Degradate(scale=self.scale,patch_size_w=patch_w,patch_size_h=patch_h)
        rotate = Rotate(degree=self.rotate_degree[randint(0,3)])
        batch_= []
        for sample in batch:
            sample_ = rotate(sample)
            sample_ = degrade(sample_)
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
        
        img_L = cv2.resize(img_H, (new_h, new_w), interpolation=cv2.INTER_CUBIC)


        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img_H = cv2.resize(img_H, (new_h, new_w), interpolation=cv2.INTER_CUBIC)



    

        return {'img_H': img_H, 'img_L': img_L}               







class AddMaskFace(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, output_size=256):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.masks = [self.line,self.rectangle,self.circle]

    def __call__(self, sample):
        img_H,img_L = sample['img_H'] , sample["img_L"]

        img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

        idx = randint(0,2)
        img_L = self.masks[idx](img_L)

        return {'img_H': img_H,'img_L': img_L}

    def line(self,image):
        s_h = randint(0,self.output_size-10)
        s_w = randint(0,self.output_size-10)
        e_h = randint(s_h,self.output_size)
        e_w = randint(s_w,self.output_size)

        img_masked = cv2.line(
            image,
            pt1 = (s_w, s_h), pt2 = (e_w, e_h),
            color = (0, 0, 0),
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
                color = (0, 0, 0),
                thickness = -1)


        return img_masked 

    def circle(self,image):
        s_h = randint(0,self.output_size-10)
        s_w = randint(0,self.output_size-10)
        raduis = randint(5,int(min(self.output_size - max(s_h,s_w),self.output_size/7)))

        img_masked = cv2.circle(
                    image,
                    center = (s_w, s_h),
                    radius = raduis,
                    color = (0, 0, 0),
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
        return {'img_H': img_H,'img_L': img_L}
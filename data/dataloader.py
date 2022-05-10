from logging import raiseExceptions
from math import fabs
import os
from cv2 import transform
import cv2
from torchvision.io import read_image
from torch.utils.data import Dataset
from utils import utils_image
import numpy as np




class Dataset(Dataset):
    def __init__(self, data_dir,transform=None):

        """data_ir = ["path_to_data/origin or path_to_data/data"] """
        
        self.origin_Images = []
        print("train with",len(data_dir),"datasets")
        for path_ in data_dir:
                self.origin_Images.extend(utils_image.get_image_paths(path_))
   

        self.transform = transform


    def __len__(self):
        return len(self.origin_Images)

    def __getitem__(self, idx):
        
        #image path
        origin_path = self.origin_Images[idx]
        if(origin_path == None):
          print("path none",self.origin_Images[idx])
          origin_path = self.origin_Images[idx-1]


        img_H = utils_image.imread_uint(origin_path, 3)# RGB H*W*C
        sample = {"img_H": img_H}

        if(self.transform != None):
            sample = self.transform(sample)

         
        return sample



    def getAugmenteddIdx(self):
        idx_augmented = []
        
        for i in range(len(self.origin_Images)):
            for j in range(self.aug_num):
                idx_augmented.append(i)
        return idx_augmented    




class FaceDataset(Dataset):
    def __init__(self, data_dir,transform=None):

        """data_ir = ["path_to_data/origin or path_to_data/data"] """
        
        self.origin_Images = []
        print("train with",len(data_dir),"datasets")
        for path_ in data_dir:
                self.origin_Images.extend(utils_image.get_image_paths(os.path.join(path_,"origin")))
   

        self.transform = transform


    def __len__(self):
        return len(self.origin_Images)

    def __getitem__(self, idx):
        
        #image path
        origin_path = self.origin_Images[idx]
        if(origin_path == None):
          print("path none",self.origin_Images[idx])
          origin_path = self.origin_Images[idx-1]


        img_H = utils_image.imread_uint(origin_path, 3)# RGB H*W*C
        sample = {"img_H": img_H}

        if(self.transform != None):
            sample = self.transform(sample)

         
        return sample



class AnimeDataSet(Dataset):
    def __init__(self, data_dir,transform=None):

        """data_ir = ["path_to_data/origin or path_to_data/data"] """
        
        self.origin_Images = []
        self.anime_Images = []
        print("train with",len(data_dir),"datasets")
        for path_ in data_dir:
                self.origin_Images.extend(utils_image.get_image_paths(os.path.join(path_,"origin")))
                self.anime_Images.extend(utils_image.get_image_paths(os.path.join(path_,"data")))


        if(len(self.anime_Images) != len(self.origin_Images)):
        
            print("data and origin folder does not have the same size")  
            print("anime",len(self.anime_Images))
            print("origin",len(self.origin_Images))   

        self.transform = transform


    def __len__(self):
        return len(self.origin_Images)

    def __getitem__(self, idx):
    
  
        #image path
        origin_path = self.origin_Images[idx]
        anime_path = self.anime_Images[idx]
        if(origin_path == None or anime_path == None):
          print("path none",self.origin_Images[idx])
          origin_path = self.origin_Images[idx-1]
          anime_path = self.anime_Images[idx-1]

        img_nameo, ext = os.path.splitext(os.path.basename(origin_path))
        img_namen, ext = os.path.splitext(os.path.basename(anime_path))

        assert img_namen  == img_nameo, 'Only data pairs with same name allowed'

        img_origin = utils_image.imread_uint(origin_path, 3)# RGB H*W*C
        img_anime = utils_image.imread_uint(anime_path, 3)# RGB H*W*C
        sample = {"img_L": img_origin,"img_H":img_anime}

        if(self.transform != None):
            sample = self.transform(sample)

         
        return sample



class ColorizeDataset(Dataset):
    def __init__(self, data_dir,transform=None):

        """data_ir = ["path_to_data/origin or path_to_data/data"] """
        
        self.origin_Images = []
        print("train with",len(data_dir),"datasets")
        for path_ in data_dir:
                self.origin_Images.extend(utils_image.get_image_paths(os.path.join(path_,"origin")))
   

        self.transform = transform


    def __len__(self):
        return len(self.origin_Images)

    def __getitem__(self, idx):
        
        #image path
        origin_path = self.origin_Images[idx]
        if(origin_path == None):
          print("path none",self.origin_Images[idx])
          origin_path = self.origin_Images[idx-1]


        img_H = utils_image.imread_uint(origin_path, 3)# RGB H*W*C
        sample = {"img_H": img_H}

        if(self.transform != None):
            sample = self.transform(sample)

         
        return sample



class GanEnhanceDataSet(Dataset):
    def __init__(self, data_dir,transform=None):

        """data_ir = ["path_to_data/origin or path_to_data/data"] """
        
        self.origin_Images = []
        print("train with",len(data_dir),"datasets")
        for path_ in data_dir:
                self.origin_Images.extend(utils_image.get_image_paths(path_))
   

        self.transform = transform


    def __len__(self):
        return len(self.origin_Images)

    def __getitem__(self, idx):
        
        #image path
        origin_path = self.origin_Images[idx]
        if(origin_path == None):
          print("path none",self.origin_Images[idx])
          origin_path = self.origin_Images[idx-1]


        img_H = utils_image.imread_uint(origin_path, 3)# RGB H*W*C
        sample = {"img_H": img_H}

        if(self.transform != None):
            sample = self.transform(sample)

         
        return sample
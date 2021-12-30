

from utils import utils_image
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
import os
import cv2
import mat
import my_lib
import sys
from mat4py import loadmat
from pymatreader import read_mat
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio 

origin_path ="/media/anis/InWork/Data/Downloaded_data/bsd/BSR/BSDS500/data/groundTruth"
dest_path  = "/media/anis/InWork/Data/dataset/BSDS500"
images_list = utils_image.get_image_paths(origin_path)
size = len(images_list)
for j in tqdm(range(size)):
    img_name, ext = os.path.splitext(os.path.basename(images_list[j]))
    # print(images_list[j])
    # img = cv2.imread(images_list[j])
    # print(img)
    # # new_img = my_lib.getImage(mat.Mat.from_array(img))
    # # new_img = np.asarray(new_img)
    # print(images_list[j])
    # data = read_mat("1.mat")
    # img = sio.loadmat(images_list[j],appendmat=True)
    # print(img)
    # # cv2.imshow('Image', img) 
    # cv2.imshow(dest_path+"/"+img_name+".jpg", img["groundTruth"])


    images = loadmat('IMAGES.mat')

    imgplot = plt.imshow(images[:,:0])
    
    plt.savefig(dest_path+"/"+img_name+".jpg")
    # plt.savefig(data["cjdata"]['image'], format='png')
    # image = loadmat(images_list[j])
    # f = h5py.File(images_list[j], 'r') #Open mat file for reading
    # cjdata = f['cjdata'] #<HDF5 group "/cjdata" (5 members)>
    # #get image member and convert numpy ndarray of type float
    # image = np.array(cjdata.get('image')).astype(np.float64) #In MATLAB: image = cjdata.image

    # label = cjdata.get('label')[0,0] #Use [0,0] indexing in order to convert lable to scalar

    # PID = cjdata.get('PID') # <HDF5 dataset "PID": shape (6, 1), type "<u2">
    # PID = ''.join(chr(c) for c in PID) #Convert to string https://stackoverflow.com/questions/12036304/loading-hdf5-matlab-strings-into-python

    # tumorBorder = np.array(cjdata.get('tumorBorder'))[0] #Use [0] indexing - convert from 2D array to 1D array.

    # tumorMask = np.array(cjdata.get('tumorMask'))

    # f.close()

    # #Convert image to uint8 (before saving as jpeg - jpeg doesn't support int16 format).
    # #Use simple linear conversion: subtract minimum, and divide by range.
    # #Note: the conversion is not optimal - you should find a better way.
    # #Multiply by 255 to set values in uint8 range [0, 255], and covert to type uint8.
    # hi = np.max(image)
    # lo = np.min(image)
    # image = (((image - lo)/(hi-lo))*255).astype(np.uint8)

    # #Save as jpeg
    # #https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
    # im = Image.fromarray(image)
    # im.save(os.path.join(dest_path,img_name+".png"))

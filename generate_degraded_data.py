from numpy.ma import add
from utils import utils_image
from utils.utils_blindsr import degradation_bsrgan,degradation_bsrgan_plus, degradation_bsrgan_plus_an
import numpy as np
import cv2
import time
import argparse
import os
from random import randint
from tqdm import tqdm
import threading
import multiprocessing as mp
import concurrent
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor



def add_noise(images_list,j,data_path):
    img_name, ext = os.path.splitext(os.path.basename(images_list[j]))
    img_H = utils_image.imread_uint(images_list[j], 3)# RGB H*W*C
    img_H = utils_image.uint2single(img_H)
    # H, W, C = img_H.shape

    # dim_ = int(min(H,W)/4)
    # if(dim_ >init_patchsize and dim_ < 512):   
    #     patch_size = randint(init_patchsize,dim_)
    
    
    img_l, img_h = degradation_bsrgan_plus(img_H, sf=1, shuffle_prob=0.5, use_sharp=False, lq_patchsize_w=img_H.shape[1],lq_patchsize_h=img_H.shape[0], isp_model=None)
    # img_l =  utils_image.single2uint(img_l)
    img_l =  utils_image.single2uint(img_l)

    # utils_image.imsave(img_l,L_path+img_name_)
    utils_image.imsave(img_l,os.path.join(data_path,'data/'+img_name+ext))

    return j


def main(data_path,workers):


    origin_path = os.path.join(data_path,"origin")
    print("process from  ",origin_path)
    images_list = utils_image.get_image_paths(origin_path)
    size = len(images_list)
    print(size)
    start = time.time()
    print("start processing....")
    with ProcessPoolExecutor(workers) as executer:
        results = [executer.submit(add_noise,images_list,j,data_path) for j in range(size)]
    #print("process time",time.time()- start)
    len_ = len(results)
    print("process time {:.3f}s per iteration".format((time.time()- start)/size))

    # idx = 0 
    # start = time.time()
    # for f  in concurrent.futures.as_completed(results):
    #     print(f.result())
    #     idx += 1
    #     if(idx % 20):
    #         print("going .... {:.2f}%  with {:.1f} s per iteration".format(idx/len_,(time.time()- start)/len_))
    

    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train_path', type=str, default="/media/anis/InWork/Data/enhance_dataset/MY_DATA/train", help='Path to high resolution images.')
    parser.add_argument('--data_valid_path', type=str, default="/media/anis/InWork/Data/dataset/URBAN/valid", help='Path to high resolution images.')
    parser.add_argument('--degradation_type', type=str, default="bsrgan_degradation", help='type of degradation.')
    parser.add_argument('--gen_images', type=int, default=16, help='number of image generated.')
    parser.add_argument('--workers', type=int, default=2, help='number of workers.')

    args = parser.parse_args()


    # p1 =mp.Process(main(args.data_train_path,))
    # p2 = mp.Process(main(args.data_valid_path,))
    # p1.start()
    # p2.start()
    # thread1 = threading.Thread(target=main,args=(args.data_train_path,))
    # thread2 = threading.Thread(target=main,args=(args.data_valid_path,))
    # thread1.start()
    # thread2.start()
    main(args.data_train_path,args.workers)
    #main(args.data_valid_path,args.workers)
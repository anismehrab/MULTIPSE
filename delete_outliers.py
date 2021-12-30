from utils import utils_image
import numpy as np
import argparse
from PIL import Image
import threading
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_train_path', type=str, default="/media/anis/InWork/Data/SuperResulotionData/superResolution_data/train", help='Path to high resolution images.')
parser.add_argument('--data_valid_path', type=str, default="/media/anis/InWork/Data/SuperResulotionData/superResolution_data/valid", help='Path to high resolution images.')

args = parser.parse_args()

def main(data_path):


    origin_path = data_path
    new_h, new_w = 1023,1023

    for i,img_path in  enumerate(utils_image.get_image_paths(origin_path)):
        image = Image.open(img_path)
        w_, h_ = image.size
        if(w_ > new_w and h_ > new_h and w_ > new_h and h_ > new_w):
            print(img_path)
        # if(w_ <new_w or h_ < new_h or w_ < new_h or h_ < new_w):
        #     print("remove"+img_path)
        #     os.remove(img_path)
        # if i% 100 ==0 :
        #     print(i)


if __name__ == '__main__':

    thread1 = threading.Thread(target=main,args=(args.data_train_path,))
    #thread2 = threading.Thread(target=main,args=(args.data_valid_path,))
    thread1.start()
    #thread2.start()

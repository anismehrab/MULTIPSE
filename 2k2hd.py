
from utils import utils_image
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--data_train_path', type=str, default="/media/anis/InWork/Data/SuperResulotionData/superResolution_data/train", help='Path to high resolution images.')
parser.add_argument('--data_valid_path', type=str, default="/media/anis/InWork/Data/SuperResulotionData/superResolution_data/valid", help='Path to high resolution images.')

args = parser.parse_args()

def main(data_path):



    origin_path = data_path+"/origin/"
    new_data_path = data_path+"/data/"
    new_h, new_w = 1056,1056
    idx = 0
    images_list = utils_image.get_image_paths(origin_path)
    size = len(images_list)
    for j in tqdm(range(size)):
        image = Image.open(images_list[j])
        w_, h_ = image.size
        if(w_ <new_w or h_ < new_h or w_ < new_h or h_ < new_w):
            print("wrong shape",image.size)
            # img_name_= str(idx)+'.png'
            # imag_c.save(new_data_path+img_name_)
            # idx += 1
            continue

        for i in range(3):
            top = np.random.randint(0, h_ - new_h)
            left = np.random.randint(0, w_ - new_w)
            imag_c = image.crop((left,top,left+new_w,top+new_h))
            img_name_= str(idx)+'.png'
            imag_c.save(new_data_path+img_name_)
            idx += 1
        


if __name__ == '__main__':

    thread1 = threading.Thread(target=main,args=(args.data_train_path,))
    thread2 = threading.Thread(target=main,args=(args.data_valid_path,))
    thread1.start()
    thread2.start()


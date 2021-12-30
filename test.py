import logging
import argparse
import torch
from models.model import BaseNet, UpsamplerNet
import os
import numpy as np
import time
from utils import utils_image,utils_logger
from utils.utils_blindsr import degradation_bsrgan_plus_an
import cv2





def main():
    parser = argparse.ArgumentParser(description='IMDN')
    parser.add_argument("--upscale_factor", type=int, default=3,
                        help='upscaling factor')
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/checkpoint_epoch_13.pth",help="checkpoint path")             
    parser.add_argument("--testset", type=str, default="test_data",help="checkpoint path")             
    args = parser.parse_args()

    utils_logger.logger_info('test_logging', log_path='test_logging.log')
    logger = logging.getLogger('test_logging')

    testsets = 'testsets'       # fixed, set path of testsets
    testset_Ls = [args.testset]

    print(torch.__version__)               # pytorch version
    print(torch.version.cuda)              # cuda version
    print(torch.backends.cudnn.version())  # cudnn version

    testsets = 'testsets'       # fixed, set path of testsets
    testset_Ls = ['test_data']#['RealSRSet']  # ['RealSRSet','DPED']
   
    base = BaseNet(upscale=args.upscale_factor)
    head = UpsamplerNet(upscale=args.upscale_factor)

    checkpoint = torch.load(args.checkpoint)
    base.load_state_dict(checkpoint["model_base_state_dict"])
    head.load_state_dict(checkpoint["model_head_state_dict"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base.eval()
    head.eval()
    base.to(device)
    head.to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for testset_L in testset_Ls:

            L_path = os.path.join(testsets, testset_L)
            #E_path = os.path.join(testsets, testset_L+'_'+model_name)
            E_path = os.path.join(testsets, testset_L+'_results_x'+str(args.upscale_factor))
            utils_image.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0
            psnr = 0
            for img in utils_image.get_image_paths(L_path):
                torch.cuda.empty_cache()
                # --------------------------------
                # (1) load img_L
                # --------------------------------
                
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                img_L = cv2.imread(img, cv2.IMREAD_COLOR)
                img_L = np.transpose(img_L if img_L.shape[2] == 1 else img_L[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB

                if((np.shape(img_L)[1] > 512 or np.shape(img_L)[2] > 512)): continue

                img_L = img_L.astype(np.float32) / 255.
                img_L = torch.from_numpy(img_L).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
                img_L = img_L.to(device)

                print("image High resoluiton size:{}".format(img_L.size()))
                # --------------------------------
                # (2) inference
                # --------------------------------
                with torch.no_grad():
                    start.record()
                    lr = base(img_L)
                    img_E = head(lr)
                    end.record()
                    torch.cuda.synchronize()
                    inf_time = start.elapsed_time(end)  # milliseconds
    
                print("image High resoluiton size:{}".format(img_E.size()))
                print("inference time: {} ms".format(inf_time))

                        
                # --------------------------------
                # (3) PSNR
                # --------------------------------
                img_E = img_E.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                img_E = (img_E * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_E = np.transpose(img_E[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                # psnr = cv2.PSNR(img_O,img_E)

                # --------------------------------
                # (4) save img_E
                # --------------------------------
                cv2.imwrite(os.path.join(E_path, img_name+'_'+"MULTISPE"+'.png'), img_E)
            # psnr = psnr / len(utils_image.get_image_paths(L_path))
            # logger.info('{:>20s} : {:f}'.format(testset_L, psnr))



if __name__ == '__main__':

    main()
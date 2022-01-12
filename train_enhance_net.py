import argparse, os
import logging
from random import randint
from numpy import float64
import torch
from torchvision import transforms
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from data.dataloader import Dataset
from data.transform import DataBatch, ToTensor,Normalize,Degradate,RandomCrop
from torch.utils.data import DataLoader
from models import enhance_model
from utils.train_utils import train,valid,save_checkpoint
from utils import utils_logger





parser = argparse.ArgumentParser()
parser.add_argument('--data_valid', nargs="+" ,default=["/media/anis/InWork/Data/enhance_dataset/DIV_FLICKR_2K/valid","/media/anis/InWork/Data/enhance_dataset/FFHQ/valid","/media/anis/InWork/Data/enhance_dataset/CAMERA_FUSION/valid","/media/anis/InWork/Data/enhance_dataset/CLOTH/valid","/media/anis/InWork/Data/enhance_dataset/DHD_CAMPUS/valid","/media/anis/InWork/Data/enhance_dataset/MY_DATA/valid","/media/anis/InWork/Data/enhance_dataset/HAND_WRITTING/valid"], help='path validation dataset.')
parser.add_argument('--data_train', nargs="+",default=["/media/anis/InWork/Data/enhance_dataset/DIV_FLICKR_2K/train","/media/anis/InWork/Data/enhance_dataset/FFHQ/train","/media/anis/InWork/Data/enhance_dataset/CAMERA_FUSION/train","/media/anis/InWork/Data/enhance_dataset/CLOTH/train","/media/anis/InWork/Data/enhance_dataset/DHD_CAMPUS/train","/media/anis/InWork/Data/enhance_dataset/MY_DATA/train","/media/anis/InWork/Data/enhance_dataset/HAND_WRITTING/train","/media/anis/InWork/Data/enhance_dataset/URBAN/train"], help='Path to trainning dataset.')

parser.add_argument("--checkpoint", type=str, default="",help="checkpoint path")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/enhance_net_checkpoints/enhance_net_x1",help="checkpoint_folder_path")
parser.add_argument("--logger_path", type=str, default="checkpoints/enhance_net_checkpoints/enhance_net_x1/train_logging.log",help="logger path")

parser.add_argument('--threads', type=int, default=4, help='threads number.')
parser.add_argument("--start_iter", type=int, default=0,help="iteration")

parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
parser.add_argument('--epoch', type=int, default=5, help='epoch.')
parser.add_argument("--scale", type=int, default=1,help="super-resolution scale")
parser.add_argument("--lr", type=float, default=1e-4,help="learning rate")
parser.add_argument("--step_size", type=int, default=50,help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=float, default=0.1,help="learning rate decay factor for step decay")

parser.add_argument("--max_dim", type=int, default=512,help="max image dimension")
parser.add_argument("--min_dim", type=int, default=128,help="min image dimension")
parser.add_argument("--max_cells", type=int, default=440*440,help="min image dimension")

args = parser.parse_args()


#init logger
utils_logger.logger_info('train_logging', log_path=args.logger_path)
logger = logging.getLogger('train_logging')
logger.info('\n')
logger.info('#################### Trainning Resumed ####################')
#load data

def reInitLoader(box):
    """box = (min_h,max_h,min_w,max_w)
        max = max_image_width * max_image_high to fit in GPU """
    data_compos = transforms.Compose([Normalize()])
    batch_compos = transforms.Compose([ToTensor()])
    dataBatch = DataBatch(transfrom=batch_compos,scale =args.scale ,max_box = box,max_cells= args.max_cells,devider=4)
    training_data = Dataset(data_dir=args.data_train,transform=data_compos)
    validation_data = Dataset(data_dir=args.data_valid,transform=data_compos)
    logger.info("===>Trainning Data:[ Train:{}  Valid:{}] Batch:{}".format(len(training_data),len(validation_data),args.batch_size))
    train_loader = DataLoader(training_data, batch_size=args.batch_size,shuffle=True, num_workers=args.threads,collate_fn=dataBatch.collate_fn)
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size,shuffle=True, num_workers=args.threads,collate_fn=dataBatch.collate_fn)
    in_ = next(iter(train_loader))
    logger.info("===>Input:[{},{},{},{}] output:[{},{},{},{}]".format(
        in_["img_L"].size(0),in_["img_L"].size(1),in_["img_L"].size(2),in_["img_L"].size(3), 
        in_["img_H"].size(0),in_["img_H"].size(1),in_["img_H"].size(2),in_["img_H"].size(3)))
    return train_loader,valid_loader

#init loaders
box = (args.min_dim,args.max_dim,args.min_dim,args.max_dim)
trainloader,validloader = reInitLoader(box)
#load models
print("loading model")
model = enhance_model.EnhanceNetSame() #enhance_model.EnhanceNet(upscale=args.scale)


#training device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
logger.info('{:>16s} : {:<s}'.format('DEVICE ID', device.type))
model.to(device)


# t = torch.cuda.get_device_properties(0).total_memory
# r = torch.cuda.memory_reserved(0)
# a = torch.cuda.memory_allocated(0)
# f = r-a  # free inside reserved
# print(a/1024)


#loass function
l1_criterion = nn.L1Loss()
l1_criterion.to(device)
logger.info("L1 loss function")

#optimzer
params = list(model.parameters()) 
optimizer = optim.Adam(params, lr=args.lr)


#load checkpoint
epoch_i = 0
if(args.checkpoint != ""):

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_base_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_i = checkpoint["epoch"] +1
    print("optimizer",optimizer)

model = [model]

#trainning

for i in range(epoch_i,epoch_i+args.epoch):

    loss_t = train(model,trainloader,optimizer,l1_criterion,i,device,args,logger)
    psnr,ssim,loss_v = valid(model,validloader,l1_criterion,device,args,logger)
    save_checkpoint(model[0],None,None,i,loss_t,loss_v,psnr,ssim,optimizer,logger,args)

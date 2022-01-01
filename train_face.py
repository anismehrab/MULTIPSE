import argparse, os
import logging
from random import randint
from numpy import float64
import torch
from torchvision import transforms
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from data.dataloader import FaceDataset
from data.transform import FaceToTensor,FaceRescale,AddMaskFace,FaceNormalize
from torch.utils.data import DataLoader
from models import model
from utils.train_utils import train,valid,save_checkpoint
from utils import utils_logger





parser = argparse.ArgumentParser()
parser.add_argument('--data_valid', nargs="+" ,default=["/media/anis/InWork/Data/face_dataset/valid"], help='path validation dataset.')
parser.add_argument('--data_train', nargs="+",default=["/media/anis/InWork/Data/face_dataset/train"], help='Path to trainning dataset.')

parser.add_argument("--checkpoint", type=str, default="",help="checkpoint path")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/face_checkpoints",help="checkpoint_folder_path")
parser.add_argument("--logger_path", type=str, default="face_train_logging.log",help="logger path")

parser.add_argument('--threads', type=int, default=4, help='threads number.')

parser.add_argument('--batch_size', type=int, default=4, help='batch size.')
parser.add_argument('--epoch', type=int, default=5, help='epoch.')
parser.add_argument("--scale", type=int, default=1,help="super-resolution scale")
parser.add_argument("--lr", type=float, default=1e-5,help="learning rate")
parser.add_argument("--step_size", type=int, default=30,help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=float, default=0.5,help="learning rate decay factor for step decay")

parser.add_argument("--max_dim", type=int, default=150,help="max image dimension")
parser.add_argument("--min_dim", type=int, default=80,help="min image dimension")
parser.add_argument("--max_cells", type=int, default=135*135,help="min image dimension")

args = parser.parse_args()


#init logger
utils_logger.logger_info('face_train_logging', log_path=args.logger_path)
logger = logging.getLogger('face_train_logging')
logger.info('\n')
logger.info('#################### Trainning Resumed ####################')
logger.info("scale"+str(args.scale))

#load data
def reInitLoader(box):
    """box = (min_h,max_h,min_w,max_w)
        max = max_image_width * max_image_high to fit in GPU """
    data_compos = transforms.Compose([FaceRescale(1024,same=True),AddMaskFace(256),FaceNormalize(),FaceToTensor()])
    training_data = FaceDataset(data_dir=args.data_train,transform=data_compos)
    validation_data = FaceDataset(data_dir=args.data_valid,transform=data_compos)
    logger.info("===>Trainning Data:[ Train:{}  Valid:{}] Batch:{}".format(len(training_data),len(validation_data),args.batch_size))
    train_loader = DataLoader(training_data, batch_size=args.batch_size,shuffle=True, num_workers=args.threads)
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size,shuffle=True, num_workers=args.threads)
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
pre_base_model = model.DownSampler_x4()
base_model = model.BaseNet()
head_model = model.UpsamplerNet_Face(upscale=args.scale)

#training device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
logger.info('{:>16s} : {:<s}'.format('DEVICE ID', device.type))
if pre_base_model != None:
    pre_base_model.to(device)
base_model.to(device)
head_model.to(device)

# t = torch.cuda.get_device_properties(0).total_memory
# r = torch.cuda.memory_reserved(0)
# a = torch.cuda.memory_allocated(0)
# f = r-a  # free inside reserved
# print(t/1024/1024)
# print(f/1024)


#loass function
l1_criterion = nn.L1Loss()
l1_criterion.to(device)
logger.info("L1 loss function")

#optimzer
params = list(head_model.parameters())
if pre_base_model != None:
    params += list(pre_base_model.parameters())

optimizer = optim.Adam(params, lr=args.lr)


#load checkpoint
epoch_i = 0
if(args.checkpoint != ""):

    checkpoint = torch.load(args.checkpoint)
    base_model.load_state_dict(checkpoint["model_base_state_dict"])
    # if pre_base_model != None:
    #     pre_base_model.load_state_dict(checkpoint["pre_model_base_state_dict"])
    # head_model.load_state_dict(checkpoint["model_head_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # epoch_i = checkpoint["epoch"] +1
    print("optimizer",optimizer)

for param in base_model.parameters():
    param.requires_grad = False

#trainning

for i in range(epoch_i,epoch_i+args.epoch):

    loss_t = train(base_model,head_model,trainloader,optimizer,l1_criterion,i,device,args,logger,pre_base_model)
    psnr,ssim,loss_v = valid(base_model,head_model,validloader,l1_criterion,device,args,logger,pre_base_model)
    save_checkpoint(base_model,head_model,i,loss_t,loss_v,psnr,ssim,optimizer,logger,args,pre_base_model)

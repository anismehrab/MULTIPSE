import argparse, os
from operator import mod
import logging
from random import randint
import torch
from torchvision import transforms
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from data.dataloader import AnimeDataSet
from data.anime_transform import AnimeNormalize,AnimeToTensor,DataBatch,AnimeTensorNormalize
from torch.utils.data import DataLoader
from models import anime_model
from utils.train_utils import train,valid,save_checkpoint,train_cuda_f16,train_with_style
from utils import utils_logger



parser = argparse.ArgumentParser()
parser.add_argument('--data_valid', nargs="+" ,default=["/media/anis/InWork/Data/anime_dataset/valid"], help='path validation dataset.')
parser.add_argument('--data_train', nargs="+",default=["/media/anis/InWork/Data/anime_dataset/train"], help='Path to trainning dataset.')

parser.add_argument("--checkpoint", type=str, default="",help="checkpoint path")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/anime_net_checkpoints/anime_net_5",help="checkpoint_folder_path")
parser.add_argument("--logger_path", type=str, default="checkpoints/anime_net_checkpoints/anime_net_5/train_logging.log",help="logger path")

parser.add_argument('--threads', type=int, default=4, help='threads number.')

parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
parser.add_argument('--epoch', type=int, default=5, help='epoch.')
parser.add_argument("--lr", type=float, default=1e-4,help="learning rate")
parser.add_argument("--step_size", type=int, default=80,help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=float, default=0.1,help="learning rate decay factor for step decay")

parser.add_argument("--max_dim", type=int, default=220,help="max image dimension")
parser.add_argument("--min_dim", type=int, default=80,help="min image dimension")
parser.add_argument("--max_cells", type=int, default=200*200,help="min image dimension") #prev 350*350

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
        
    batch_compos = transforms.Compose([AnimeNormalize(),AnimeToTensor()])
    dataBatch = DataBatch(transfrom=batch_compos,max_box = box,max_cells= args.max_cells,devider=4,forc_size=None,add_style=True)
    training_data = AnimeDataSet(data_dir=args.data_train)
    validation_data = AnimeDataSet(data_dir=args.data_valid)
    logger.info("===>Trainning Data:[ Train:{}  Valid:{}] Batch:{}".format(len(training_data),len(validation_data),args.batch_size))
    train_loader = DataLoader(training_data, batch_size=args.batch_size,shuffle=True, num_workers=args.threads,collate_fn=dataBatch.collate_fn)
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size,shuffle=True, num_workers=args.threads,collate_fn=dataBatch.collate_fn)
    in_ = next(iter(train_loader))
    logger.info("===>Input:[{},{},{},{}] output:[{},{},{},{}]".format(
        in_["img_L"].size(0),in_["img_L"].size(1),in_["img_L"].size(2),in_["img_L"].size(3), 
        in_["img_H"].size(0),in_["img_H"].size(1),in_["img_H"].size(2),in_["img_H"].size(3)))
    return train_loader,valid_loader




    return train_loader,valid_loader

#init loaders
box = (args.min_dim,args.max_dim,args.min_dim,args.max_dim)
trainloader,validloader = reInitLoader(box)
#load models
print("loading model")
model = anime_model.AnimeNet4()


#training device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
logger.info('{:>16s} : {:<s}'.format('DEVICE ID', device.type))
model.to(device)

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
optimizer = optim.Adam(model.parameters(), lr=args.lr)


#load checkpoint
epoch_i = 0
if(args.checkpoint != ""):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_base_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_i = checkpoint["epoch"] +1
    print("optimizer",optimizer)


# model.half()
# for layer in model.modules():
#     if isinstance(layer,nn.BatchNorm2d):
#         layer.float()
# for param in base_model.parameters():
#     param.requires_grad = False

#trainning

for i in range(epoch_i,epoch_i+args.epoch):

    loss_t = train_with_style([model],trainloader,optimizer,l1_criterion,i,device,args,logger)
    psnr,ssim,loss_v = valid([model],validloader,l1_criterion,device,args,logger)
    save_checkpoint(model,None,None,i,loss_t,loss_v,psnr,ssim,optimizer,logger,args)

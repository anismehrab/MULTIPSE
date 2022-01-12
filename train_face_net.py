import argparse, os
import logging
from random import randint
import torch
from torchvision import transforms
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from data.dataloader import FaceDataset
from data.face_transforms import FaceToTensor,AddMaskFace,FaceNormalize,DataBatch
from torch.utils.data import DataLoader
from models import face_model
from utils.train_utils import train,valid,save_checkpoint
from utils import utils_logger





parser = argparse.ArgumentParser()
parser.add_argument('--data_valid', nargs="+" ,default=["/media/anis/InWork/Data/face_dataset/valid"], help='path validation dataset.')
parser.add_argument('--data_train', nargs="+",default=["/media/anis/InWork/Data/face_dataset/train"], help='Path to trainning dataset.')

parser.add_argument("--checkpoint", type=str, default="",help="checkpoint path")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/face_net_checkpoints/previous_checkpoits",help="checkpoint_folder_path")
parser.add_argument("--logger_path", type=str, default="checkpoints/face_net_checkpoints/previous_checkpoits/face_net_train_logging.log",help="logger path")

parser.add_argument('--threads', type=int, default=4, help='threads number.')

parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
parser.add_argument('--epoch', type=int, default=5, help='epoch.')
parser.add_argument("--lr", type=float, default=1e-4,help="learning rate")
parser.add_argument("--step_size", type=int, default=80,help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=float, default=0.1,help="learning rate decay factor for step decay")

parser.add_argument("--max_dim", type=int, default=512,help="max image dimension")
parser.add_argument("--min_dim", type=int, default=128,help="min image dimension")
parser.add_argument("--max_cells", type=int, default=420*420,help="min image dimension")
parser.add_argument("--mask_color", type=str ,required=True,help="mask color: black or white")

args = parser.parse_args()


#init logger
utils_logger.logger_info('face_net_train_logging', log_path=args.logger_path)
logger = logging.getLogger('face_net_train_logging')
logger.info('\n')
logger.info('#################### Trainning Resumed ####################')

#load data
def reInitLoader(box):
    """box = (min_h,max_h,min_w,max_w)
        max = max_image_width * max_image_high to fit in GPU """
    if(args.mask_color == "black"):
        mask_transfrom = AddMaskFace(color=(0,0,0))
    elif(args.mask_color == "while"):
        mask_transfrom = AddMaskFace(color=(255,255,255))
        
    batch_compos = transforms.Compose([mask_transfrom,FaceNormalize(),FaceToTensor()])
    dataBatch = DataBatch(transfrom=batch_compos,max_box = box,max_cells= args.max_cells)
    training_data = FaceDataset(data_dir=args.data_train)
    validation_data = FaceDataset(data_dir=args.data_valid)
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
model = face_model.FaceNet()


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

# for param in base_model.parameters():
#     param.requires_grad = False

#trainning

for i in range(epoch_i,epoch_i+args.epoch):

    loss_t = train([model],trainloader,optimizer,l1_criterion,i,device,args,logger)
    psnr,ssim,loss_v = valid([model],validloader,l1_criterion,device,args,logger)
    save_checkpoint(model,None,None,i,loss_t,loss_v,psnr,ssim,optimizer,logger,args)

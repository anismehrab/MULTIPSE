import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import utils
import skimage.color as sc
import time



def train(models_list,train_loader,optimizer,l1_criterion,epoch,device,args,logger):

    for model in models_list:
        model.train()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    logger.info('===> TrainEpoch ={}  lr = {}'.format(epoch,optimizer.param_groups[0]['lr']))
    l_loss = 0.0
    iteration = 0
    start.record()
    for sample in train_loader:

        torch.cuda.empty_cache()
        lr_tensor = sample["img_L"].to(device)  # ranges from [0, 1]
        hr_tensor = sample["img_H"].to(device)  # ranges from [0, 1]
        
        optimizer.zero_grad()

        temp_tensor = lr_tensor;
        for model in models_list:
            temp_tensor = model(temp_tensor)

        # torch.cuda.empty_cache()
        #print(temp_tensor.size())

        loss_l1 = l1_criterion(temp_tensor, hr_tensor)
        l_loss += loss_l1.item()
        
        loss_l1.backward()
        optimizer.step()
        iteration += 1
        if iteration % 200 == 1:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(train_loader),l_loss/iteration))
    
        
    end.record()
    torch.cuda.synchronize()
    l_loss = l_loss/len(train_loader)
    logger.info("===> Epoch[{}]: Loss_l1: {:.5f}  Duration: {:<5f} min".format(epoch,l_loss,start.elapsed_time(end)/60000.0))
    return 



def valid(models_list,valid_loader,l1_criterion,device,args,logger):
    for model in models_list:
        model.eval()
    avg_psnr, avg_ssim = 0, 0
    l_loss = 0.0
    for sample in valid_loader:
        torch.cuda.empty_cache()
        lr_tensor = sample["img_L"].to(device)  # ranges from [0, 1]
        hr_tensor = sample["img_H"].to(device)  # ranges from [0, 1]

        
        with torch.no_grad():
            temp_tensor = lr_tensor
            for model in models_list:
                temp_tensor = model(temp_tensor)


        loss_l1 = l1_criterion(temp_tensor, hr_tensor)
        l_loss += loss_l1.item()

        temp_psnr = 0
        tem_ssim = 0
        batch = temp_tensor.size()[0]
        for i in range(batch):
            sr_img = utils.tensor2np(temp_tensor.detach()[i])
            gt_img = utils.tensor2np(hr_tensor.detach()[i])
            temp_psnr += utils.compute_psnr(sr_img, gt_img)
            tem_ssim += utils.compute_ssim(sr_img, gt_img)
        temp_psnr = temp_psnr / batch
        tem_ssim = tem_ssim / batch

        avg_psnr += temp_psnr
        avg_ssim += tem_ssim
    avg_psnr,avg_ssim,l_loss = avg_psnr/len(valid_loader),avg_ssim/len(valid_loader),l_loss/len(valid_loader)
    logger.info("===> Valid. psnr: {:.4f}, ssim: {:.4f}, loss: {:.4f}".format(avg_psnr, avg_ssim,l_loss))
    return avg_psnr,avg_ssim,l_loss



def save_checkpoint(base_model,head_model,pre_base_model,epoch,loss_train,loss_valid,psnr,ssim,optimizer,logger,args):
    model_foler = args.checkpoint_path
    model_path = ""
    if not os.path.exists(model_foler):
        os.makedirs(model_foler)
    if(pre_base_model != None):    
        model_path = os.path.join(model_foler,"checkpoint_" + "epoch_{}.pth".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_base_state_dict': base_model.state_dict(),
            'pre_model_base_state_dict': pre_base_model.state_dict(),
            'model_head_state_dict': head_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'psnr': psnr,
            'ssim':ssim
            }, model_path)
    elif(base_model != None and head_model != None):
        model_path = os.path.join(model_foler,"checkpoint_" + "epoch_{}.pth".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_base_state_dict': base_model.state_dict(),
            'model_head_state_dict': head_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'psnr': psnr,
            'ssim':ssim
            }, model_path)

    elif(base_model != None):
        model_path = os.path.join(model_foler,"checkpoint_base" + "epoch_{}.pth".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_base_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'psnr': psnr,
            'ssim':ssim
            }, model_path)
    else:
        model_path = os.path.join(model_foler,"checkpoint_head" + "epoch_{}.pth".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_head_state_dict': head_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'psnr': psnr,
            'ssim':ssim
            }, model_path)

    logger.info("===> Checkpoint saved to {}".format(model_path))
    logger.info('\n')


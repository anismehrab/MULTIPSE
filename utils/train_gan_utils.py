import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import utils
import skimage.color as sc
import time



def train(gen,disc,train_loader,gen_opt,disc_opt,adv_criterion,recon_criterion,epoch,device,args,logger,lambda_recon = 100,display_step = 200):
    
    scaler = torch.cuda.amp.GradScaler()

    gen.train()
    disc.train()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    utils.adjust_learning_rate(gen_opt, epoch, args.step_size, args.lr, args.gamma)
    utils.adjust_learning_rate(disc_opt, epoch, args.step_size, args.lr, args.gamma)
    logger.info('===> TrainEpoch ={}  lr = {}'.format(epoch,gen_opt.param_groups[0]['lr']))
    generator_loss = 0
    discriminator_loss = 0
    iteration = 0
    start.record()
    for sample in train_loader:

        lr_tensor = sample["img_L"].to(device)  # ranges from [0, 1]
        hr_tensor = sample["img_H"].to(device)  # ranges from [0, 1]
        
         ### Update discriminator ###
        disc_opt.zero_grad() # Zero out the gradient before backpropagation
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                fake = gen(lr_tensor)
            disc_fake_hat = disc(fake.detach(), lr_tensor) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(hr_tensor, lr_tensor)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
        
        # disc_loss.backward(retain_graph=True) # Update gradients
        # disc_opt.step() # Update optimizer
        scaler.scale(disc_loss).backward(retain_graph=True)
        scaler.step(disc_opt)
        scaler.update()

        torch.cuda.empty_cache()
        ### Update generator ###
        gen_opt.zero_grad()
        with torch.cuda.amp.autocast():
            fake = gen(lr_tensor)
            disc_fake_hat = disc(fake, lr_tensor)
            gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
            gen_rec_loss = recon_criterion(hr_tensor, fake)
            gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
        # gen_loss.backward() # Update gradients
        # gen_opt.step() # Update optimizer
        scaler.scale(gen_loss).backward()
        scaler.step(gen_opt)
        scaler.update()




        # Keep track of the average discriminator loss
        discriminator_loss += disc_loss.item()
        # Keep track of the average generator loss
        generator_loss += gen_loss.item()


        iteration += 1
        if iteration % display_step == 1:
            # print(f"Epoch: {epoch} Step: {iteration}/{len(train_loader)} Generator loss: {generator_loss/iteration}, Discriminator loss: {discriminator_loss/iteration}")
            print("===> Epoch[{}]: Step: {}/{} Generator loss: {:.5f},Discriminator loss: {:.5f}".format(epoch,iteration,len(train_loader),generator_loss,discriminator_loss))
        
    end.record()
    torch.cuda.synchronize()
    discriminator_loss = discriminator_loss/len(train_loader)
    generator_loss = generator_loss/len(train_loader)
    logger.info("===> Epoch[{}]: Generator loss: {:.5f},Discriminator loss: {:.5f}  Duration: {:<5f} min".format(epoch,generator_loss,discriminator_loss,start.elapsed_time(end)/60000.0))
    return generator_loss , discriminator_loss




def valid(model,valid_loader,l1_criterion,device,args,logger):

    model.eval()
    avg_psnr, avg_ssim = 0, 0
    l_loss = 0.0
    for sample in valid_loader:
        torch.cuda.empty_cache()
        lr_tensor = sample["img_L"].to(device)  # ranges from [0, 1]
        hr_tensor = sample["img_H"].to(device)  # ranges from [0, 1]
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out_tensor = model(lr_tensor)
            loss_l1 = l1_criterion(out_tensor, hr_tensor)
            l_loss += loss_l1.item()

        temp_psnr = 0
        tem_ssim = 0
        batch = out_tensor.size()[0]
        for i in range(batch):
            sr_img = utils.tensor2np(out_tensor.detach()[i])
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



def save_checkpoint(generator,discriminator,i,gen_loss,disc_loss,loss_v,psnr,ssim,logger,args):
    model_foler = args.checkpoint_path
    model_path = ""
    if not os.path.exists(model_foler):
        os.makedirs(model_foler)
     
    model_path = os.path.join(model_foler,"checkpoint_" + "epoch_{}.pth".format(i))
    torch.save({
        'epoch': i,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_loss': gen_loss,
        'disc_loss': disc_loss,
        'valid_loss':loss_v,
        'psnr': psnr,
        'ssim':ssim
        }, model_path)

    logger.info("===> Checkpoint saved to {}".format(model_path))
    logger.info('\n')

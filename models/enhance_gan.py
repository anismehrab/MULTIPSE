import torch.nn as nn
from . import imdb as B
import torch
from models.model import UpsamplerNet,BaseNet
from models.gan_utils import *



class GenEnhanceV1(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,act_type='relu',cc_acti='relu'):
        super(GenEnhanceV1, self).__init__()

        upscale=4
        self.downsampleer = nn.Sequential(
                    B.conv_block(in_nc,int(nf/upscale), kernel_size=3, stride=2,act_type=act_type),
                    B.conv_block(int(nf/upscale), nf, kernel_size=3, stride=2,act_type=act_type),
                    B.conv_layer(nf, nf, kernel_size=3),
                    )

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB2 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB3 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB4 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB5 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB6 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB7 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB8 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB9 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB10 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB11 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)
        self.IMDB12 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=cc_acti)

        num_modules=12
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        # print("input",input.size())
        out_fea = self.downsampleer(input)
        # print("out_fea",out_fea.size())

        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)
        out_B7 = self.IMDB7(out_B6)
        out_B8 = self.IMDB8(out_B7)       
        out_B9 = self.IMDB9(out_B8)
        out_B10 = self.IMDB10(out_B9)
        out_B11 = self.IMDB11(out_B10)       
        out_B12 = self.IMDB12(out_B11) 
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12], dim=1))
        #print("out_B",out_B.size())        

        out_lr = torch.add(self.LR_conv(out_B), out_fea)
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output        


class GanDiscriminatorV1(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    
    def __init__(self, in_nc=6, nf=32):
        super(GanDiscriminatorV1, self).__init__()
        self.upfeature = FeatureMapBlock(in_nc, nf)
        self.contract1 = ContractingBlock(nf, use_bn=False) 
        self.contract2 = ContractingBlock(nf * 2)
        self.contract3 = ContractingBlock(nf * 4)
        self.contract4 = ContractingBlock(nf * 8)
        #### START CODE HERE ####
        self.final = nn.Conv2d(nf * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn ,x2,x3,x4

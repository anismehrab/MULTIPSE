import torch.nn as nn
from . import imdb as B
import torch

class BaseNet(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=12):
        super(BaseNet, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.IMDB6 = B.IMDModule(in_channels=nf)
        self.IMDB7 = B.IMDModule(in_channels=nf)
        self.IMDB8 = B.IMDModule(in_channels=nf)
        self.IMDB9 = B.IMDModule(in_channels=nf)
        self.IMDB10 = B.IMDModule(in_channels=nf)
        self.IMDB11 = B.IMDModule(in_channels=nf)
        self.IMDB12 = B.IMDModule(in_channels=nf)
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.fun = nn.quantized.FloatFunctional()


    def forward(self, input):
        #print("input",input.size())
        out_fea = self.fea_conv(input)
        #print("out_fea",out_fea.size())

        out_B1 = self.IMDB1(out_fea)
        #print("out_B1",out_B1.size())

        out_B2 = self.IMDB2(out_B1)
        #print("out_B2",out_B2.size())
        out_B3 = self.IMDB3(out_B2)
        #print("out_B3",out_B3.size())
        out_B4 = self.IMDB4(out_B3)
        #print("out_B4",out_B4.size())
        out_B5 = self.IMDB5(out_B4)
        #print("out_B5",out_B5.size())
        out_B6 = self.IMDB6(out_B5)
        #print("out_B6",out_B6.size())        
        out_B7 = self.IMDB7(out_B6)
        out_B8 = self.IMDB8(out_B7)       
        out_B9 = self.IMDB9(out_B8)
        out_B10 = self.IMDB10(out_B9)
        out_B11 = self.IMDB11(out_B10)
        out_B12 = self.IMDB12(out_B11)
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12], dim=1))
        #print("out_B",out_B.size())        

        out_lr = self.LR_conv(out_B) + out_fea
        
        return out_lr



class UpsamplerNet(nn.Module):
    def __init__(self, nf=64,out_nc=3, upscale=4):
        super(UpsamplerNet, self).__init__()

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
    
    def forward(self,input):
        output = self.upsampler(input)
        return output






class BaseNet_Large(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=20):
        super(BaseNet_Large, self).__init__()
        self.upscale = upscale
        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, 3),
                                      nn.ReLU(inplace=True),
                                      B.conv_layer(nf, nf, 3, stride=2, bias=False),)

        self.block1 = B.IMDModule_Large(nf)
        self.block2 = B.IMDModule_Large(nf)
        self.block3 = B.IMDModule_Large(nf)
        self.block4 = B.IMDModule_Large(nf)
        self.block5 = B.IMDModule_Large(nf)
        self.block6 = B.IMDModule_Large(nf)
        self.LR_conv = B.conv_layer(nf, nf, 1, bias=False)


    def forward(self, input):

        fea = self.fea_conv(input)
        out_b1 = self.block1(fea)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b5 = self.block5(out_b4)
        out_b6 = self.block6(out_b5)

        out_lr = self.LR_conv(out_b6) + fea
        return out_lr        




class DownSampler_x4(nn.Module):
    def __init__(self, nf=3,in_nc=3):
        super(DownSampler_x4, self).__init__()

        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, kernel_size=3, stride=2),
                                      nn.ReLU(),
                                      B.conv_layer(nf, nf, kernel_size=3),
                                      nn.ReLU(),
                                      B.conv_layer(nf, nf, kernel_size=3),
                                      nn.ReLU(),
                                      B.conv_layer(nf, nf, kernel_size=3, stride=2),
                                      nn.ReLU(),
                                      B.conv_layer(nf, nf, kernel_size=3),
                                      nn.ReLU(),
                                      B.conv_layer(nf, nf, kernel_size=3),
                                      nn.ReLU())
    
    def forward(self,input):
        output = self.fea_conv(input)
        return output



class UpsamplerNet_Face(nn.Module):
    def __init__(self, nf=64,out_nc=3, upscale=4):
        super(UpsamplerNet_Face, self).__init__()

        upsample_block = B.pixelshuffle_block_large
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
    
    def forward(self,input):
        output = self.upsampler(input)
        return output
import torch.nn as nn
from . import imdb as B
import torch

class EnhanceNet(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,upscale=4):
        super(EnhanceNet, self).__init__()



        #down by x2
        pre_upscale = 2
        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, int(nf/pre_upscale), kernel_size=3, stride=2),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/pre_upscale), int(nf/pre_upscale), kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/pre_upscale), int(nf/pre_upscale), kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/pre_upscale), nf, kernel_size=3),
                    )

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


        num_modules=12
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        #up by 
      
        self.upsampler = nn.Sequential(
                                    B.conv_layer(nf, out_nc * (upscale ** 2) * (pre_upscale ** 2), kernel_size=3),
                                    nn.PixelShuffle(pre_upscale),
                                    B.conv_layer(out_nc * (upscale ** 2), out_nc * (upscale ** 2), kernel_size=3),
                                    nn.PixelShuffle(upscale)
                                    )



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

        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output

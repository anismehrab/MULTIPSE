import torch.nn as nn
from . import imdb as B
import torch


class EnhanceNet(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,upscale = 4):
        super(EnhanceNet, self).__init__()

        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, int(nf/upscale), kernel_size=3, stride=2),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/upscale), int(nf/upscale), kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/upscale), int(nf/upscale), kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/upscale), nf, kernel_size=3, stride=2),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(nf, nf, kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05))

        self.conv_fea = B.conv_layer(nf, nf, kernel_size=1);
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

        num_modules=10
        upscale_factor = upscale * 4
        out_channels = out_nc*(upscale_factor)**2
        self.conv_cat = B.conv_block(nf * num_modules, out_channels, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(out_channels, out_channels, kernel_size=3)

        self.upsampler = nn.Sequential(B.conv_layer(out_channels, out_channels, kernel_size=3),
                                    nn.PixelShuffle(upscale_factor)) 

    def forward(self, input):
        # print("input",input.size())
        down_sample = self.downsampleer(input)
        out_fea = self.conv_fea(down_sample)
        # print("out_fea",out_fea.size())
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_s_1 = torch.add(out_B2 , out_fea)
        out_B3 = self.IMDB3(out_s_1)
        out_B4 = self.IMDB4(out_B3)
        out_s_2 = torch.add(out_B4 , out_s_1)
        out_B5 = self.IMDB5(out_s_2)
        out_B6 = self.IMDB6(out_B5)
        out_s_3 = torch.add(out_B6 , out_s_2)
        out_B7 = self.IMDB7(out_s_3)
        out_B8 = self.IMDB8(out_B7)       
        out_s_3 = torch.add(out_B8 , out_s_3)
        out_B9 = self.IMDB9(out_s_3)

        out_B = self.conv_cat(torch.cat([out_fea,out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9], dim=1))
        #print("out_B",out_B.size())        

        out_lr = self.LR_conv(out_B)
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output



class EnhanceNetSame(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3):
        super(EnhanceNetSame, self).__init__()

        upscale=4
        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, int(nf/upscale), kernel_size=3, stride=2),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/upscale), int(nf/upscale), kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/upscale), int(nf/upscale), kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(int(nf/upscale), nf, kernel_size=3, stride=2),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(nf, nf, kernel_size=3),
                    B.activation('lrelu', neg_slope=0.05),
                    B.conv_layer(nf, nf, kernel_size=3),
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

        num_modules=9
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        # print("input",input.size())
        out_fea = self.downsampleer(input)
        # print("out_fea",out_fea.size())

        out_B1 = self.IMDB1(out_fea)
        # print("out_B1",out_B1.size())

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

        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9], dim=1))
        #print("out_B",out_B.size())        

        out_lr = torch.add(self.LR_conv(out_B), out_fea)
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output        
import torch.nn as nn
from . import imdb as B
import torch


class EnhanceNet(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,upscale = 4,act_type = "relu"):
        super(EnhanceNet, self).__init__()

        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, int(nf), kernel_size=3, stride=2),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf), int(nf), kernel_size=3),
                    B.activation(act_type, neg_slope=0.05),
                    )

        self.conv_fea = B.conv_layer(nf, nf, kernel_size=3);
        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB2 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB3 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB4 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB5 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB6 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB7 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB8 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB9 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB10 = B.IMDModule(in_channels=nf,act_type=act_type)
        self.IMDB11 = B.IMDModule(in_channels=nf,act_type=act_type)

        num_modules=11
        upscale_factor = upscale * 2
        out_channels = out_nc*(upscale_factor)**2
        self.conv_cat = B.conv_block(nf * num_modules, out_channels, kernel_size=1, act_type=act_type)
        # self.LR_conv = B.conv_layer(out_channels, out_channels, kernel_size=3)
        
        self.upsampler = nn.Sequential(nn.PixelShuffle(upscale_factor)) 
        # self.upsampler = nn.Sequential(B.conv_layer(out_channels, out_channels, kernel_size=3),
        #                             nn.PixelShuffle(upscale_factor)) 

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
        out_s_2 = torch.add(out_B4 , out_B2)
        out_B5 = self.IMDB5(out_s_2)
        out_B6 = self.IMDB6(out_B5)
        out_s_3 = torch.add(out_B6 , out_B4)
        out_B7 = self.IMDB7(out_s_3)
        out_B8 = self.IMDB8(out_B7)       
        out_s_3 = torch.add(out_B8 , out_B6)
        out_B9 = self.IMDB9(out_s_3)
        out_B10 = self.IMDB10(out_B9)
        out_s_4 = torch.add(out_B10 , out_B8)
        out_B11 = self.IMDB11(out_s_4)
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11], dim=1))
        #print("out_B",out_B.size())        

        # out_lr = self.LR_conv(out_B)
        output = self.upsampler(out_B)
        # print("out_put",output.size())

        return output



class EnhanceNetX1(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,act_type="relu"):
        super(EnhanceNetX1, self).__init__()

        upscale=4
        self.downsampleer = nn.Sequential(
                    B.conv_block(in_nc,int(nf/upscale), kernel_size=3, stride=2,act_type=act_type),
                    B.conv_block(int(nf/upscale),int(nf/upscale), kernel_size=3, stride=1,act_type=act_type),
                    B.conv_block(int(nf/upscale),nf, kernel_size=3, stride=1,act_type=act_type),
                    B.conv_block(nf, nf, kernel_size=3, stride=2,act_type=act_type),
                    B.conv_block(nf, nf, kernel_size=3, stride=1,act_type=act_type),
                    B.conv_layer(nf, nf, kernel_size=3),
                    )

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB2 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB3 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB4 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB5 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB6 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB7 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB8 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB9 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB10 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)

        num_modules=10
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
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10], dim=1))
        #print("out_B",out_B.size())        

        out_lr = torch.add(self.LR_conv(out_B), out_fea)
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output        


class EnhanceNetX2(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,act_type  ="relu"):
        super(EnhanceNetX2, self).__init__()

        
        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, nf, kernel_size=3, stride=2),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(nf, nf, kernel_size=3)
                    )

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB2 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB3 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB4 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB5 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB6 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB7 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB8 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB9 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)

        num_modules=9
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        
    
        upsample_block = B.pixelshuffle_block
        self.upsampler_1 = upsample_block(nf, 16, upscale_factor=2)
        self.upsampler_2 = upsample_block(16, out_nc, upscale_factor=2)


    def forward(self, input):
        # print("input",input.size())
        out_fea = self.downsampleer(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)
        out_B7 = self.IMDB7(out_B6)
        out_B8 = self.IMDB8(out_B7)       
        out_B9 = self.IMDB9(out_B8)

        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9], dim=1))
        #print("out_B",out_B.size())        

        out_lr = torch.add(self.LR_conv(out_B), out_fea)
        ur = self.upsampler_1(out_lr)
        output = self.upsampler_2(ur)
        # print("out_put",output.size())

        return output                

class EnhanceNetX3(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,act_type = "relu"):
        super(EnhanceNetX3, self).__init__()

        
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
                   
        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB2 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB3 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB4 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB5 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB6 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB7 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB8 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB9 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB10 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB11 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)
        self.IMDB12 = B.IMDModule(in_channels=nf,act_type=act_type,cc_acti=act_type)


        num_modules=12
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        
        upscale=3
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        # print("input",input.size())
        out_fea = self.fea_conv(input)
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





class EnhanceNetX4(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3):
        super(EnhanceNetX4, self).__init__()

        
        self.fea_conv = B.conv_layer(in_nc, int(nf), kernel_size=3)
                   

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
        
        upscale=4
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        # print("input",input.size())
        out_fea = self.fea_conv(input)
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
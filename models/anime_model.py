from audioop import bias
import torch.nn as nn
from . import imdb as B
import torch

class AnimeNet(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3):
        super(AnimeNet, self).__init__()

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
        self.IMDB10 = B.IMDModule(in_channels=nf)
        self.IMDB11 = B.IMDModule(in_channels=nf)
        self.IMDB12 = B.IMDModule(in_channels=nf)
        self.IMDB13 = B.IMDModule(in_channels=nf)
        self.IMDB14 = B.IMDModule(in_channels=nf)
        num_modules=14
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
        out_B10 = self.IMDB10(out_B9)
        out_B11 = self.IMDB11(out_B10)       
        out_B12 = self.IMDB12(out_B11)    
        out_B13 = self.IMDB13(out_B12)
        out_B14 = self.IMDB14(out_B13)       
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12, out_B13, out_B14], dim=1))
        #print("out_B",out_B.size())        

        out_lr = torch.add(self.LR_conv(out_B), out_fea)
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output


class AnimeNet2(nn.Module):
    def __init__(self, in_nc=3, nf=128,out_nc=3,bias = True,act_type = 'relu'):
        super(AnimeNet2, self).__init__()

        upscale=4
        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, int(nf/4), kernel_size=3,bias=bias),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf/4), int(nf/2), kernel_size=3, stride=2,bias=bias),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf/2), int(nf/2), kernel_size=3,bias=bias),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf/2), nf, kernel_size=3, stride=2,bias=bias),
                    )

        # IMDBs
        self.IMDB1 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB2 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB3 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB4 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB5 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB6 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB7 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB8 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB9 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB10 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB11 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB12 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)

        num_modules=12
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type,bias=True)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3,bias=True)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        # print("input",input.size())
        out_fea = self.downsampleer(input)
        # print("out_fea",out_fea.size())

        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2+out_fea)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5+out_B2)
        out_B7 = self.IMDB7(out_B6)
        out_B8 = self.IMDB8(out_B7)       
        out_B9 = self.IMDB9(out_B8+out_B5)
        out_B10 = self.IMDB10(out_B9)
        out_B11 = self.IMDB11(out_B10)       
        out_B12 = self.IMDB12(out_B11+out_B8)    
    
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12], dim=1))
        #print("out_B",out_B.size())        

        out_lr = torch.add(self.LR_conv(out_B), out_fea)
        output = self.upsampler(out_lr)
        # print("out_put",output.size())

        return output



class AnimeNet3(nn.Module):
    def __init__(self,  in_nc=3, nf=96, out_nc=3,act_type ="relu"):
        super(AnimeNet3, self).__init__()
        upscale=2
        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, int(nf/2), 3,bias=False),
                                      B.activation(act_type=act_type),
                                      B.conv_layer(int(nf/2), nf, 3, stride=2,bias=False))

        self.block1 = B.IMDModule_Large(nf,distillation_rate=0.25,act_type=act_type)
        self.block2 = B.IMDModule_Large(nf,distillation_rate=0.25,act_type=act_type)
        self.block3 = B.IMDModule_Large(nf,distillation_rate=0.25,act_type=act_type)
        self.block4 = B.IMDModule_Large(nf,distillation_rate=0.25,act_type=act_type)
        self.block5 = B.IMDModule_Large(nf,distillation_rate=0.25,act_type=act_type)
        self.block6 = B.IMDModule_Large(nf,distillation_rate=0.25,act_type=act_type)

        self.conv_cat = B.conv_block(nf * 6, nf, kernel_size=1, act_type=act_type)

        self.LR_conv = B.conv_layer(nf, nf, 3, bias=True)

        self.upsampler = nn.Sequential(B.conv_layer(nf, out_nc * ((upscale) ** 2), 3,bias=True),
                                         nn.PixelShuffle(upscale))
        
    def forward(self, input):

        fea = self.fea_conv(input)
        out_b1 = self.block1(fea)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b5 = self.block5(out_b4)
        out_b6 = self.block6(out_b5)
        out_B = self.conv_cat(torch.cat([out_b1, out_b2, out_b3, out_b4, out_b5, out_b6], dim=1))

        out_lr = self.LR_conv(out_B) + fea

        output = self.upsampler(out_lr)

        return output





class AnimeNet4(nn.Module):
    def __init__(self, in_nc=3, nf=128,out_nc=3,bias = True,act_type = 'relu'):
        super(AnimeNet4, self).__init__()

        upscale=4
        self.downsampleer = nn.Sequential(
                    B.conv_layer(in_nc, int(nf/4), kernel_size=3,bias=bias),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf/4), int(nf/2), kernel_size=3, stride=2,bias=bias),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf/2), int(nf/2), kernel_size=3,bias=bias),
                    B.activation(act_type, neg_slope=0.05),
                    B.conv_layer(int(nf/2), nf, kernel_size=3, stride=2,bias=bias),
                    )

        # IMDBs
        self.IMDB1 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB2 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB3 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB4 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB5 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB6 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB7 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB8 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB9 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB10 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB11 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB12 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB13 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB14 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB15 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB16 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB17 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB18 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB19 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        self.IMDB20 = B.IMDModule_(in_channels=nf,act_type = act_type,bias=bias)
        num_modules=20
        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type,bias=True)
        self.LR_conv = B.conv_layer(nf, int(nf/2), kernel_size=3,bias=True)
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.UR_conv1 = B.conv_block(int(nf/2), int(nf/4), kernel_size=3,bias=True,act_type=act_type)
        self.UR_conv2 = B.conv_layer(int(nf/4), 3, kernel_size=3,bias=True)



    def forward(self, input):
        # print("input",input.size())
        out_fea = self.downsampleer(input)
        # print("out_fea",out_fea.size())

        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2+out_fea)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5+out_B2)
        out_B7 = self.IMDB7(out_B6)
        out_B8 = self.IMDB8(out_B7)       
        out_B9 = self.IMDB9(out_B8+out_B5)
        out_B10 = self.IMDB10(out_B9)
        out_B11 = self.IMDB11(out_B10)       
        out_B12 = self.IMDB12(out_B11+out_B8)    
        out_B13 = self.IMDB13(out_B12)
        out_B14 = self.IMDB14(out_B13)
        out_B15 = self.IMDB15(out_B14+out_B11)
        out_B16 = self.IMDB16(out_B15)       
        out_B17 = self.IMDB17(out_B16)
        out_B18 = self.IMDB18(out_B17+out_B14)
        out_B19 = self.IMDB19(out_B18)       
        out_B20 = self.IMDB20(out_B19)  
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12, out_B13, out_B14, out_B15, out_B16, out_B17, out_B18, out_B19, out_B20], dim=1))
        #print("out_B",out_B.size())        
        
        out_lr = self.LR_conv(out_B)
        out_ur1 = self.upsampler(out_lr)
        out_conv1 = self.UR_conv1(out_ur1)
        out_ur2 = self.upsampler(out_conv1)
        out_conv1 = self.UR_conv2(out_ur2)

        return out_conv1


class AnimeNet5(nn.Module):
    def __init__(self, in_nc=3, nf=64,out_nc=3,act_type = 'relu'):
        super(AnimeNet5, self).__init__()

        self.downsampleer_O = nn.Sequential(
                    B.conv_block(in_nc,int(nf/2),3,stride=2,act_type=act_type),
                    B.conv_block(int(nf/2),int(nf/2),3,stride=1,act_type=act_type),
                    B.conv_layer(int(nf/2), nf, kernel_size=3, stride=2)
                    )

        self.downsampleer_A = nn.Sequential(
                    B.conv_block(in_nc,int(nf/2),3,stride=2,act_type=act_type),
                    B.conv_block(int(nf/2),int(nf/2),3,stride=1,act_type=act_type),
                    B.conv_layer(int(nf/2), nf, kernel_size=3, stride=2)
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
        self.IMDB10 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB11 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB12 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB13 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        self.IMDB14 = B.IMDModule(in_channels=nf,cc_acti=act_type,act_type=act_type)
        num_modules=14

        self.conv_cat = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        
    
        upsample_block = B.pixelshuffle_block
        self.upsampler_1 = upsample_block(nf, 16, upscale_factor=2)
        self.upsampler_2 = upsample_block(16, out_nc, upscale_factor=2)
        

    def forward(self, input):
        # print("input",input.size())
        out_fea = self.downsampleer_O(input[0])
        out_or  = self.downsampleer_A(input[1])

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
        out_B13 = self.IMDB13(out_B12)
        out_B14 = self.IMDB14(out_B13)       
        out_B = self.conv_cat(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_B11, out_B12, out_B13, out_B14], dim=1))
         


        out_lr = torch.add(self.LR_conv(out_B), out_or)
        ur = self.upsampler_1(out_lr)
        output = self.upsampler_2(ur)


        return output
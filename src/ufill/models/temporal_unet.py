import torch
import torch.nn as nn

from .resnet import resnet50
from .vgg import VGG16
import numpy as np

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg1    = VGG16(pretrained = pretrained, in_channels=20)
            self.vgg2    = VGG16(pretrained = pretrained, in_channels=17)
            # self.vgg3    = VGG16(pretrained = pretrained, in_channels=18)
            # self.vgg3    = VGG16(pretrained = pretrained, in_channels=2)

            # , in_channels=6   #修改input的通道数，或者时候个数
            # in_filters  = [192, 384, 768, 1024]256
            # in_filters  = [384, 768, 1536, 2048]512
            in_filters  = [512, 1024, 2048, 4096]#按照feat通道和out通道对应相加得到

        elif backbone == "resnet50":
            self.resnet1 = resnet50(pretrained = pretrained, in_channels=4)
            self.resnet2 = resnet50(pretrained = pretrained, in_channels=1)
            # in_filters  = [192, 512, 1024, 3072]
            in_filters  = [384, 1024, 2048, 6144]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        # out_filters = [64, 128, 256, 512]
        out_filters = [128, 256, 512,1024]
        # out_filters = [256, 512, 1024,2048]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            input1 =[]
            input2 = []
            input3=[]
            input1=inputs[:,:4,:,:]#GEMS
            input2=inputs[:,4,:,:]#GEOS
            input2=input2[:,np.newaxis,:,:]
            input3=inputs[:,5:21,:,:]#ps,day_wei
            input33=inputs[:,21:38,:,:]#pre_VCD_TROPOMI,ps,pre_day_wei
            input44=inputs[:,38:55,:,:]#nex_VCD_TROPOMI,ps,nex_day_wei

            input11 = torch.cat([input1, input3], dim=1)#GEMS,ps,day_wei
            [feat11, feat12, feat13, feat14, feat15] = self.vgg1.forward(input11)
            input22 = torch.cat([input2, input3], dim=1)#GEOS,ps,day_wei
            [feat21, feat22, feat23, feat24, feat25] = self.vgg2.forward(input22)

            # input33 = torch.cat([input4, input3], dim=1)
            [feat31, feat32, feat33, feat34, feat35] = self.vgg2.forward(input33)

            # input44 = torch.cat([input4, input3], dim=1)
            [feat41, feat42, feat43, feat44, feat45] = self.vgg2.forward(input44)


            # [feat31, feat32, feat33, feat34, feat35] = self.vgg3.forward(input3)

           #cat两组特征向量，并修改网络input和output通道数
            feat1= torch.cat([feat11, feat21, feat31, feat41],1)
            feat2= torch.cat([feat12, feat22, feat32, feat42],1)
            feat3= torch.cat([feat13, feat23, feat33, feat43],1)
            feat4= torch.cat([feat14, feat24, feat34, feat44],1)
            feat5= torch.cat([feat15, feat25, feat35, feat45],1)


        elif self.backbone == "resnet50":
            input1 =[]
            input2 = []
            input1=inputs[:,:4,:,:]
            [feat11, feat12, feat13, feat14, feat15] = self.resnet1.forward(input1)
            input2=inputs[:,4,:,:]
            input2=input2[:,np.newaxis,:,:]
            [feat21, feat22, feat23, feat24, feat25] = self.resnet2.forward(input2)
           #cat两组特征向量，并修改网络input和output通道数
            feat1= torch.cat([feat11, feat21],1)
            feat2= torch.cat([feat12, feat22],1)
            feat3= torch.cat([feat13, feat23],1)
            feat4= torch.cat([feat14, feat24],1)
            feat5= torch.cat([feat15, feat25],1)

            

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            backbones = (self.vgg1, self.vgg2)
        elif self.backbone == "resnet50":
            backbones = (self.resnet1, self.resnet2)
        for backbone in backbones:
            for param in backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            backbones = (self.vgg1, self.vgg2)
        elif self.backbone == "resnet50":
            backbones = (self.resnet1, self.resnet2)
        for backbone in backbones:
            for param in backbone.parameters():
                param.requires_grad = True

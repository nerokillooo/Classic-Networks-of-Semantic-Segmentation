import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


#Build Bilinear_interpolation
def Bilinear_interpolation(src, new_size):# Input original size and new size of image
    dst_w, dst_h = new_size # width and height of target image
    src_h, src_w = src.shape[:2]# width and height of original image
    # return the original size if size of target image is same with original one
    if src_h == dst_h and src_w == dst_w:
        return src.copy()

    # if new size is different with original
    # Calculate the zoom factor of target image
    scale_x = float(src_w)/dst_w
    scale_y = float(src_h)/dst_h

    # generate an empty image whose size is same with target
    dst = np.zeros((dst_h, dst_w, 3), dtype = np.unit8) # 3 is number of channels

    # Traverse every coordinate in each channel through loop
    for n in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # get central point
                src_x = (dst_x+0.5)*scale_x-0.5
                src_y = (dst_y+0.5)*scale_y-0.5

                # get up-left point
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))

                # get up-right point and < boundry 
                src_x_1 = min(src_x_0+1, src_w-1)
                src_y_1 = min(src_y_0+1, src_h-1)

                # 双线性插值
                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst

class Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

def make_layers(in_channels, layer_list):
    """
    in_channels: 3the number of input's channels
    layer_list: as : [64,64]
    """
    layers = []
    for v in layer_list:
        layers += [Block(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)
        
    def forward(self, x):
        out = self.layer(x)
        return out

class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        #VGG19bn model
        # original spatial size is 224*224
        self.layer1 = Layer(3, [64, 64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #112
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #56
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #28
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #14
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Through 5 pooling, the spatial size is 7*7

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=3)
        self.dropout1 = nn.Dropout(0.85)

        self.conv7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.85)

        self.conv8 = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1)

        
    def forward(self, x):
        x = self.pool1(self.layer1(x))
            
        x = self.pool2(self.layer2(x))
 
        x = self.pool3(self.layer3(x))
        s3 = x # 1/8
            
        x = self.pool4(self.layer4(x))
        s4 = x # 1/16
            
        x = self.pool5(self.layer5(x))
        s5 = x # 1/32
        # use Bilinear interpolation to upsample the coarse outputs
        s5 = Bilinear_interpolation(s5, (n_class, n_class))
        s4 += s5

        
        s4 = Bilinear_interpolation(s4, (n_class, n_class))
        s3 += s4

        s3 = Bilinear_interpolation(s3, (n_class, n_class))

        return s3
            
                 
    

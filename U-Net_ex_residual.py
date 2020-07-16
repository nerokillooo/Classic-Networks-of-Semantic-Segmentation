import torch
import torch.nn as nn

class downConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm):
        super(downConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = int(padding)))
        block.append(nn.ReLU())

        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))

        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = int(padding)))
        block.append(nn.ReLU())

        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))

        self.downblock = nn.Sequential(*block)
    def forward(self, x):
        identity = x
        
        out = self.downblock(x)
        # residual block
        out += identity
        return out

class upConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, padding, batch_norm):
        super(upConvblock, self).__init__()
        if up_mode =='upconv':
            self.up = nn.Convtranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)#value of stride need to do test

        elif up_mode=='upsample':
            self.up = nn.Sequential(
                nn.upsample('bilinear', scale_factor = 2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.upblock = upConvBlock(in_channels, out_channels, padding, batch_norm)


    def crop(self, layer,target_size):
        _,_,layer_height,layer_width = layer.size()
        diff_y = (layer_height - target_size[0])//2
        diff_x = (layer_width - target_size[1])//2
        return layer[:,:,diff_y:(diff_y+target_size[0]),diff_x:(diff_x+target_size[1])]
        

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.crop(bridge, up.shape[2:])
        out = torch.cat((up, crop1),1)
        out = self.upblock(out)
        return out

class UNet(nn.Module):
    def __init__(
        self,
        in_channels = 1,
        n_class = 2,# number of final classification
        depth = 5,#depth of the network
        wf = 6, # channels of each layer is 2**wf, 1st layer is 2**6 = 64
        padding = False,
        batch_norm = False,
        up_mode = 'upconv',
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(downConvBlock(prev_channels, 2**(wf+i),padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth-i)):
            self.up_path.append(upConvBlock(prev_channels, 2**(wf+i), padding, batch_norm))
            prev_channels = 2**(wf+i)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)#Conv1x1
                                
    def forward(self,x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.max_pool2d(x,2)
        for i,up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
            return self.last(x)

x = torch.randn((1,1,572,572))
unet = UNet()
unnet.eval()
y_unet = unet(x)

y_unet.size()
                          
        
        

        
    

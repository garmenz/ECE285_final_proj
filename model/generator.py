import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class Generator(nn.Module):
    def __init__(self, inputc, outputc):
        '''
        Generator is Encoder-Decoder structure

        inputc: number of input channel
        outputc: number of output channel
        '''
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(inputc, 64, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn1_d = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn2_d = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn3_d = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=2)
        self.bn4_d = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)
        self.bn5_d = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
        self.bn6_d = nn.BatchNorm2d(128)
        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.bn7_d = nn.BatchNorm2d(64)
        self.deconv8 = nn.ConvTranspose2d(64, outputc, kernel_size=4, padding=1, stride=2)
        self.bn8_d = nn.BatchNorm2d(outputc)
        self.dropout = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        x: expected shape N*C*256*256
        '''
        # encoder
        x = self.conv1(x)  # shape: N * 64 * 128 * 128
        # x = self.relu(self.bn1(self.conv1(x)))  # shape: N * 64 * 128 * 128
        x = self.relu(self.bn2(self.conv2(x)))  # shape: N * 128 * 64 * 64
        x = self.relu(self.bn3(self.conv3(x)))  # shape: N * 256 * 32 * 32
        x = self.relu(self.bn4(self.conv4(x)))  # shape: N * 512 * 16 * 16
        x = self.relu(self.bn5(self.conv5(x)))  # shape: N * 512 * 8 * 8
        x = self.relu(self.bn6(self.conv6(x)))  # shape: N * 512 * 4 * 4
        x = self.relu(self.bn7(self.conv7(x)))  # shape: N * 512 * 2 * 2
        x = self.relu(self.bn8(self.conv8(x)))  # shape: N * 512 * 1 * 1

        # decoder
        x = self.relu(self.dropout(self.bn1_d(self.deconv1(x))))    # shape: N * 512 * 2 * 2
        x = self.relu(self.dropout(self.bn2_d(self.deconv2(x))))    # shape: N * 512 * 4 * 4
        x = self.relu(self.dropout(self.bn3_d(self.deconv3(x))))    # shape: N * 512 * 8 * 8
        x = self.relu(self.bn4_d(self.deconv4(x)))    # shape: N * 512 * 16 * 16
        x = self.relu(self.bn5_d(self.deconv5(x)))   # shape: N * 256 * 32 * 32
        x = self.relu(self.bn6_d(self.deconv6(x)))    # shape: N * 128 * 64 * 64
        x = self.relu(self.bn7_d(self.deconv7(x)))   # shape: N * 64 * 128 * 128 
        x = self.relu(self.bn8_d(self.deconv8(x)))    # shape: N * outputC * 256 * 256
        x = self.tanh(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, inputc, outputc):
        super(UNetGenerator, self).__init__()
        self.conv1 = nn.Conv2d(inputc, 64, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn1_d = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn2_d = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn3_d = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn4_d = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(1024, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn5_d = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(512, 128, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn6_d = nn.BatchNorm2d(128)
        self.deconv7 = nn.ConvTranspose2d(256, 64, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn7_d = nn.BatchNorm2d(64)
        self.deconv8 = nn.ConvTranspose2d(128, outputc, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn8_d = nn.BatchNorm2d(outputc)
        self.dropout = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        x: expected shape N*C*256*256
        '''
        # encoder
        x1 = self.relu(self.conv1(x))  # shape: N * 64 * 128 * 128
        # x = self.relu(self.bn1(self.conv1(x)))  # shape: N * 64 * 128 * 128
        x2 = self.relu(self.bn2(self.conv2(x1)))  # shape: N * 128 * 64 * 64
        x3 = self.relu(self.bn3(self.conv3(x2)))  # shape: N * 256 * 32 * 32
        x4 = self.relu(self.bn4(self.conv4(x3)))  # shape: N * 512 * 16 * 16
        x5 = self.relu(self.bn5(self.conv5(x4)))  # shape: N * 512 * 8 * 8
        x6 = self.relu(self.bn6(self.conv6(x5)))  # shape: N * 512 * 4 * 4
        x7 = self.relu(self.bn7(self.conv7(x6)))  # shape: N * 512 * 2 * 2
        x8 = self.relu(self.bn8(self.conv8(x7)))  # shape: N * 512 * 1 * 1

        # decoder
        x_1 = self.relu(self.dropout(self.bn1_d(self.deconv1(x8))))    # shape: N * 512 * 2 * 2
        x_1 = torch.cat((x_1, x7), 1)                                   # shape: N * 1024 * 2 * 2
        x_2 = self.relu(self.dropout(self.bn2_d(self.deconv2(x_1))))    # shape: N * 512 * 4 * 4
        x_2 = torch.cat((x_2, x6), 1)                                   # shape: N * 1024 * 4 * 4
        x_3 = self.relu(self.dropout(self.bn3_d(self.deconv3(x_2))))    # shape: N * 512 * 8 * 8
        x_3 = torch.cat((x_3, x5), 1)                                   # shape: N * 1024 * 8 * 8
        x_4 = self.relu(self.bn4_d(self.deconv4(x_3)))                  # shape: N * 512 * 16 * 16
        x_4 = torch.cat((x_4, x4), 1)                                   # shape: N * 1024 * 16 * 16
        x_5 = self.relu(self.bn5_d(self.deconv5(x_4)))                  # shape: N * 256 * 32 * 32
        x_5 = torch.cat((x_5, x3), 1)                                   # shape: N * 512 * 32 * 32
        x_6 = self.relu(self.bn6_d(self.deconv6(x_5)))                  # shape: N * 128 * 64 * 64
        x_6 = torch.cat((x_6, x2), 1)                                   # shape: N * 256 * 64 * 64
        x_7 = self.relu(self.bn7_d(self.deconv7(x_6)))                  # shape: N * 64 * 128 * 128 
        x_7 = torch.cat((x_7, x1), 1)                                   # shape: N * 128 * 128 * 128
        x_8 = self.relu(self.deconv8(x_7))                              # shape: N * outputC * 256 * 256
        x = self.tanh(x)
        return x

    
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        self.model = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
#         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         # gradually reduce the number of filters from ngf * 8 to ngf
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)




# https://github.com/viktor-ktorvi/1d-convolutional-neural-networks

import torch.nn as nn
from utils.pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D
from utils.sizes import Conv1dLayerSizes, TransposeConv1dLayerSizes


class Network2(nn.Module):
    def __init__(self, downsample ,signal_len, kerneln, channeln):
        super(Network2, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)

        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            #nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network3(nn.Module):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network3, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)


        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
           # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.convout(x)
        x = self.up(x)
        return x

class Network5(nn.Module):
    def __init__(self, downsample, signal_len, kerneln, channeln):
        super(Network5, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)


        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            #nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.convout(x)
        x = self.up(x)
        return x

class Network7(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network7, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.convout(x)
        x = self.up(x)
        return x





class Network9(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network9, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
           # nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.convout(x)
        x = self.up(x)
        return x




class Network11(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network11, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
          #  nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.convout(x)
        x = self.up(x)
        return x




class Network13(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network13, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            #nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.convout(x)
        x = self.up(x)
        return x


class Network15(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network15, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            #nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.convout(x)
        x = self.up(x)
        return x



class Network23(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network23, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv15 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv17 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv19 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv20 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv21 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )

        

    
        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            #nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.convout(x)
        x = self.up(x)
        return x





class Network35(nn.Module):
    def __init__(self, downsample,  signal_len, kerneln, channeln):
        super(Network35, self).__init__()

        # encoder
        self.down=PixelUnshuffle1D(downsample)
        self.up=PixelShuffle1D(downsample)

        
        convint_sizes = Conv1dLayerSizes(in_len=signal_len, in_ch=downsample, out_ch=channeln, kernel=kerneln, padding=kerneln//2 )

        self.convint = nn.Sequential(
            nn.Conv1d(in_channels=convint_sizes.in_ch, out_channels=convint_sizes.out_ch, kernel_size=convint_sizes.kernel_size, padding=convint_sizes.padding),
            nn.ReLU(),
            
        )

        conv2_sizes = Conv1dLayerSizes(in_len=convint_sizes.out_len, in_ch=convint_sizes.out_ch, out_ch=convint_sizes.out_ch, kernel=kerneln, padding=kerneln//2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv15 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv17 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv18 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv19 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv20 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv21 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv24 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv25 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv26 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv27 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv28 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv29 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv30 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv31 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
        self.conv33 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch, out_channels=conv2_sizes.out_ch, kernel_size=conv2_sizes.kernel_size, padding=conv2_sizes.padding),
            nn.BatchNorm1d(convint_sizes.out_ch),
            nn.ReLU(),
        )
    
        convout_sizes = Conv1dLayerSizes(in_len=conv2_sizes.out_len, in_ch=conv2_sizes.out_ch, out_ch=downsample, kernel=kerneln, padding=kerneln//2) 
        self.convout = nn.Sequential(
            nn.Conv1d(in_channels=convout_sizes.in_ch, out_channels=convout_sizes.out_ch, kernel_size=convout_sizes.kernel_size, padding=convout_sizes.padding),
            #nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        x = self.down(x)
        x = self.convint(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)
        x = self.conv26(x)
        x = self.conv27(x)
        x = self.conv28(x)
        x = self.conv29(x)
        x = self.conv30(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv33(x)
        x = self.convout(x)
        x = self.up(x)
        return x
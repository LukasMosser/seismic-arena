"""
Modified from:

MIT License

Copyright (c) 2020 Alex Shenfield

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn

from torchvision.transforms import CenterCrop


# basic convolutional block for the unet model
class UNetConvolutionalBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        
        super().__init__()
        
        # conv 3x3 -> relu -> conv 3x3 -> relu 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu_ = nn.ReLU()
        
        # we are using "same" padding here - rather than the unpadded version they
        # use in the paper. this means we don't have to mess about with cropping
        # the ground truth masks before calculating the loss
        
        # note, as in the original paper, we are not applying any normalisation
        
    
    # do the forward pass through the UNetConvolutionalBlock
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu_(out)
        out = self.conv2(out)
        out = self.relu_(out)
        return out


# u-net encoder module
class Encoder(nn.Module):
    
    def __init__(self, enc_channels=[3, 64, 128, 256, 512, 1024], dropout:float=0.5):
        
        super().__init__()
        
        # get all the convolutional blocks in the encoder network
        self.enc1 = UNetConvolutionalBlock(enc_channels[0], enc_channels[1])
        self.enc2 = UNetConvolutionalBlock(enc_channels[1], enc_channels[2])
        self.enc3 = UNetConvolutionalBlock(enc_channels[2], enc_channels[3])
        self.enc4 = UNetConvolutionalBlock(enc_channels[3], enc_channels[4])
        self.enc5 = UNetConvolutionalBlock(enc_channels[4], enc_channels[5])
        
        # get the pooling layer that will follow each convolutional block
        self.pool = nn.MaxPool2d(2, 2)
        
        # also, we need some dropout (according to the paper)
        self.dropout = nn.Dropout(dropout)
        
      
    # do the forward pass through the UNet encoder (getting the feature output
    # at each stage)
    def forward(self, x):
        
        e1 = self.enc1(x)
        
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        
        e4 = self.pool(e3)
        e4 = self.enc4(e4)
        e4 = self.dropout(e4)
        
        e5 = self.pool(e4)
        e5 = self.enc5(e5)
        e5 = self.dropout(e5)
        
        return [e1, e2, e3, e4, e5]
    
    
# u-net decoder module
class Decoder(nn.Module):
    
    def __init__(self, dec_channels=[1024, 512, 256, 128, 64]):
        
        super().__init__()
        
        self.channels = dec_channels
        
        # get the up-convolutions
        up_list = [
            nn.ConvTranspose2d(dec_channels[i], dec_channels[i+1], 2, 2) \
            for i in range(len(dec_channels) - 1)
        ]
        self.upconvs  = nn.ModuleList(up_list)
                
        # get all the convolutional blocks in the decoder network
        block_list = [
            UNetConvolutionalBlock(dec_channels[i], dec_channels[i+1]) \
            for i in range(len(dec_channels) - 1)
        ]
        self.decoder_blocks = nn.ModuleList(block_list)
        
        
    # do the forward pass through the UNet decoder. here we upsample (through 
    # up-convolutions), concatenate the feature map from the encoder, and 
    # apply our basic convolutional block
    def forward(self, x, encoder_features):
        
        for i in range(len(self.channels) - 1):            
            x = self.upconvs[i](x)
            y = self.crop(encoder_features[i], x)
            x = torch.cat([x, y], dim=1)
            x = self.decoder_blocks[i](x)
            
        return x
                        
        
    # take the centre crop of the feature map
    def crop(self, features, x):
        
        _, _, h, w = x.shape
        features = CenterCrop([h, w])(features)
        return features
    
    
# put everything together to form the unet model
class UNet(nn.Module):
    
    def __init__(self, num_classes=1,
                 enc_channels=[3, 64, 128, 256, 512, 1024],
                 dec_channels=[1024, 512, 256, 128, 64], dropout:float=0.5):
        
        super().__init__()
        self.encoder = Encoder(enc_channels, dropout=dropout)
        self.decoder = Decoder(dec_channels)
               
        # i think the head should be more complicated than this (?)
        self.head    = nn.Conv2d(dec_channels[-1], num_classes, 1)           
        
        # actually - having looked at the original unet - this seems fine. they
        # use two outputs and a softmax loss function (which is incorporated 
        # into the loss function in the pytorch code)
        
    def forward(self, x):
        features = self.encoder(x)
        out      = self.decoder(features[::-1][0], features[::-1][1:])
        out      = self.head(out)
        return out
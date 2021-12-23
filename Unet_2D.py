## U-net

# import numpy as np
import torch.nn as nn
import torch
# from torchvision import transforms
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from PIL import Image 

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
 
     
# enc_Block = Block(1,64)
# enc_Block(torch.randn(1,1,128,128)).shape
  

class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
  
    
# encoder = Encoder()
# # # input image
# x    = torch.randn(1, 1, 572, 572)
# ftrs = encoder(x)

# for ftr in ftrs: print(ftr.shape)


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            # enc_ftrs = encoder_features[i]
            enc_ftrs = self.center_crop(encoder_features[i], (x.shape[2],x.shape[3]))
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def center_crop(self, img, output_size):
        v = img.shape
        th, tw = output_size
        i = int(round((v[2] - th) / 2.))
        j = int(round((v[3] - tw) / 2.))
        return img[:,:,i:i+th,j:j+tw]
    
    # def crop(self, enc_ftrs, x):
    #     _, _, H, W = x.shape
    #     # enc_ftrs = torchvision.transforms.CenterCrop([H,W])(enc_ftrs)
    #     # transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop((H,W)), transforms.ToTensor()])
    #     # enc_ftrs = transform(enc_ftrs)
    #     enc_ftrs = center_crop(enc_ftrs,(50,50))
    #     return enc_ftrs
    

# decoder = Decoder()
# x = torch.randn(1, 1024, 28, 28)
# decoder(x, ftrs[::-1][1:]).shape


class UNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=True, out_sz=(512,512)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
    
    
# unet = UNet()
# x    = torch.randn(1, 1, 512, 512)
# out = unet(x)

# plt.figure()
# plt.imshow(np.squeeze(x.numpy()),cmap='gray')
# plt.figure()
# plt.imshow(out[0,0,:,:].detach().cpu().numpy(),cmap='gray')

# x.shape
# out.shape

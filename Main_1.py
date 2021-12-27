import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import glob

import load_data
import Unet_2D


class DataLoader():
    def __init__(self, path_data, pat):      
        path_data = os.path.normpath(path_data)
        self.pat_list = glob.glob(os.path.normpath( path_data + "\*raw.nii.gz"))
        self.data_list = []
        self.mask_list = []
        
        self.slice = []
        self.n_pat = []
        self.size_data=[]
        # print(self.pat_list[0])
        # self.name = glob.glob(self.pat_list[0] + "/*nii.gz")
        # print(' '.join(self.name[0]))
        
        for i in pat:
            path = os.path.normpath(self.pat_list[i])
            
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(path)
            file_reader.ReadImageInformation()
            size=file_reader.GetSize()
    
            ind = np.random.permutation(np.arange(20,size[2])) 
            
            for k in range(10):
                # for k in [0]:
                self.data_list = self.data_list + [os.path.normpath(path)]
                p =  os.path.normpath(path.replace("raw","mask",1))
                self.mask_list = self.mask_list + [p]
                self.slice = self.slice + [ind[k]]
                self.n_pat = self.n_pat + [i]
                self.size_data = self.size_data + [size]
                # print(' '.join(self.data_list[i]))
                # print(self.path)
                
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        size = self.size_data[index]
        size_cut = [224,224,1];
        vel_maxTr = [np.maximum(size[0]-size_cut[0]-1,1), np.maximum(size[1]-size_cut[1]-1,1)]
        
        transl = [np.random.randint(0,vel_maxTr[0]), np.random.randint(0,vel_maxTr[1]), int(self.slice[index])]

        img = load_data.read_nii_position(self.data_list[index], size_cut, transl)
        img = torch.tensor(((img.astype(np.float32))+1024)/4096)
        # img = img.unsqueeze(0)
        
        size_cut = [224,224,1];
        mask = load_data.read_nii_position(self.mask_list[index], size_cut, transl)
        mask[mask<0]=0
        mask[mask>0]=255
        mask = torch.tensor(mask.astype(np.bool_))
        # mask = img.unsqueeze(0)
        
        return img, mask



torch.cuda.empty_cache()

#training loader
# loaderTr = DataLoader(path_data = "C:\Data\Verse2019\data_reload", pat=range(0,5))
loaderTr = DataLoader(path_data = "C:\Data\Verse2019\data_reload", pat=[0,1,3,4,5])
trainloader = data.DataLoader(loaderTr,batch_size=2, num_workers=0, shuffle=True, drop_last=True)

# testing loader
loaderTe = DataLoader(path_data = "C:\Data\Verse2019\data_reload", pat=range(6,8))
testloader = data.DataLoader(loaderTe,batch_size=2, num_workers=0, shuffle=False, drop_last=True)


##### create NET --  U-net

net = Unet_2D.UNet(enc_chs=(1,64,128,256), dec_chs=(256,128, 64), out_sz=(224,224))
net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-8)


#### Training
it = 0
train_loss = []
test_loss = []
train_acc = []
test_acc = []

for epoch in range(50):
    
    acc_tmp = []
    loss_tmp = []
    
    for k,(img,mask) in enumerate(trainloader):
        it+=1

        img = img.unsqueeze(1)
        mask = mask.unsqueeze(1)
        img = img.cuda()
        mask = mask.cuda()
        
        output = net(img)
    
        output = torch.sigmoid(output)
        loss = dice_loss(mask,output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mask_num=mask.detach().cpu().numpy()>0.5
        clas=output.detach().cpu().numpy()>0.5
        acc=np.mean((clas==mask_num))
        
        loss_tmp.append(loss.detach().cpu().numpy())
        acc_tmp.append(acc)
        
    train_loss.append(np.mean(loss_tmp))
    train_acc.append(np.mean(acc_tmp))
    
    torch.cuda.empty_cache()

    acc_tmp = []
    loss_tmp = []
    for kk,(img, mask) in enumerate(testloader):
        # with torch.no_grad():
    
        img = img.unsqueeze(1)
        mask = mask.unsqueeze(1)
        img=img.cuda()
        mask=mask.cuda()
    
        output=net(img)
    
        output = torch.sigmoid(output)
        loss = dice_loss(mask,output)
    
    
        mask_num=mask.detach().cpu().numpy()>0.5
        clas=output.detach().cpu().numpy()>0.5
        acc=np.mean((clas==mask_num))
    
        acc_tmp.append(acc)
        loss_tmp.append(loss.cpu().detach().numpy())
    
    test_loss.append(np.mean(loss_tmp))
    test_acc.append(np.mean(acc_tmp))

    plt.close()
    plt.figure()
    plt.plot(train_loss,color='red')
    plt.plot(test_loss,color='blue')
    plt.show()

    torch.cuda.empty_cache()

torch.cuda.empty_cache()

# output = net(data)

# output = torch.sigmoid(output)
# loss = dice_loss(lbl,output)



# x = torch.randn(1, 1, 512, 512)


img, mask = loaderTr[0]
img = img.unsqueeze(0).unsqueeze(0)

out = net(img.cuda())
# out = net(x)
       
plt.figure()
plt.imshow(np.squeeze(img[0,0,:,:].detach().cpu().numpy()),cmap='gray')
plt.figure()
plt.imshow(np.squeeze(mask.detach().cpu().numpy()),cmap='gray')
plt.figure()
plt.imshow(out[0,0,:,:].detach().cpu().numpy(),cmap='gray')
plt.show()


###### testovani funkcnosti
# img, mask = loader[3]

# plt.figure()
# plt.imshow(img,cmap='gray')
# plt.figure()
# plt.imshow(mask,cmap='gray')


####### blbosti na testovani

# path_data = r"C:\Data\Verse2020\verse122\verse122.nii.gz"
# img, H = medpy.io.load(path_data)
# img = img[:,:,100]
# img = cv2.resize(img, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
# img = torch.tensor(((img)+1024)/4096)

# img1 = img.reshape(1,168,168)
# img = np.res(img,(64,64))
# img.resize((28,28))
# img = im.resize((28,28))

# plt.figure()
# plt.imshow(img,cmap='gray')

# path_mask = r"C:\Data\Verse2020\verse122\verse122_seg.nii.gz"
# mask, Hm = medpy.io.load(path_mask)

# plt.figure()
# plt.imshow(img[:,:,100],cmap='gray')
# plt.show()
# plt.imshow(mask[:,:,100],cmap='gray')
# plt.show()

# fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
# ax1.imshow(montage(img), cmap ='bone')
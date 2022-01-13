clear all
close all
clc

path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Data\patient001\' ;

filename = 'patient001_frame01.nii.gz';
filenameM = 'patient001_frame01_gt.nii.gz';

data = niftiread([path_data, '/', filename]);
mask = niftiread([path_data, '/', filenameM]);


rect=[40,70,128,128];
data = data(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4),:,:);
mask = mask(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4),:,:);

data = mat2gray(data);

data = (data - 0 ) ./ (0.8 - 0);
data(data>1)=1;

mask = mat2gray(mask);
% mask = (mask - 0 ) ./ (3 - 0);
% colormap = jet
% mask = gray2ind(mask)
% mask = permute(mask,[1,2,4,3]);
% mask = ind2rgb(  mask, colormap);


% imshow5(data)
% imshow5(mask)

path_save = pwd;

fps = 4;
name = 'ACDC_3D_ED';
IMG = permute(data,[1,2,4,3]);
save_gif(IMG, name, path_save, fps)

fps = 4;
name = 'ACDC_3D_ED_mask';
IMG = permute(mask,[1,2,4,3]);
save_gif(IMG, name, path_save, fps)


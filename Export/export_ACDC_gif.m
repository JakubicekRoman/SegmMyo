clear all
close all
clc

path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Data\patient001\' ;

filename = 'patient001_4d.nii.gz';

data = niftiread([path_data, '/', filename]);

rect=[40,70,128,128];
data = data(rect(1):rect(1)+rect(3),rect(2):rect(2)+rect(4),:,:);


data = mat2gray(data);

data = (data - 0 ) ./ (0.4 - 0);
data(data>1)=1;

% imshow5(data)

path_save = pwd;

fps = 10;
name = 'ACDC_4D';
IMG = data(:,:,4,:);
save_gif(IMG, name, path_save, fps)


% 
% fps = 4;
% name = 'ACDC_3D_ED';
% IMG = data(:,:,:,1);
% IMG = permute(IMG,[1,2,4,3]);

% fps = 4;
% name = 'ACDC_3D_ES';
% IMG = data(:,:,:,12);
% IMG = permute(IMG,[1,2,4,3]);
% 
% 
% save_gif(IMG, name, path_save, fps)



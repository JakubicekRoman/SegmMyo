
clear all
close all
clc

path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Data\A018_\' 

D1 = dir([path_data, 'series0019-Body\*.dcm'])
D2 = dir([path_data, 'series0030-Body\*.dcm'])
D3 = dir([path_data, 'series0063-Body\*.dcm'])

data=[];
RECT = [30,60,80,80];
for i = 1:14
    img = dicomread([D1(i).folder, '\', D1(i).name ]);
    img = imcrop(img, RECT );
    data(:,:,1,i) = img;
    img = dicomread([D2(i).folder, '\', D2(i).name ]);
    img = imcrop(img, RECT );
    data(:,:,2,i) = img;
    img = dicomread([D3(i).folder, '\', D3(i).name ]);
    img = imcrop(img, RECT );
    data(:,:,3,i) = img;
end

mean(data,[1,2,4])

data = data ./ permute([1500,3000,1500],[3,1,2,4] );

% imshow5(data)

path_save = pwd;
fps = 4;

name = 't1pre';
save_gif(data(:,:,1,:), name, path_save, fps)
name = 't2';
save_gif(data(:,:,2,:), name, path_save, fps)
name = 't1post';
save_gif(data(:,:,3,:), name, path_save, fps)


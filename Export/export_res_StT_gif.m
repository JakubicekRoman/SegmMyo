clear all
close all
clc

% path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Programs\MyoSeg\exported_images\T1_pre\' 
path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Programs\MyoSeg\exported_images\T1_post\' 
% path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Programs\MyoSeg\exported_images\T2\' 

path_save = 'C:\Users\NB-UBMI-786\Desktop\VUT\Programs\MyoSeg\exported_images\gifs';

D = dir([path_data '\*.png'])

data=[];
RECT = [112,35,329-112,249-35];

% pat = 'A025'
pat = 'A036'

slice = '0';

p=1;
for i = 1:length(D) 
    s = split(D(i).name,'_');
    if contains(s{2},pat)
        img = imread([D(i).folder, '\', D(i).name ]);
        img = imcrop(img, RECT );

%         img = histeq(img);
        img = uint8(((double(img)./255) - 0.0) / (0.7 - 0.0).*255);
%         img = uint8((im2double(img).^0.7).*255);

        data(:,:,:,p) = img;
        p = p+1;
    end
end

imshow(img)

fps = 4;

name = ['res_StT_T1_post_' pat '_' slice];
save_gif(uint8(data), name, path_save, fps)



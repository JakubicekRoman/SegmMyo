clear all
close all
clc

% path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Programs\MyoSeg\example_4D\' 
path_data = 'C:\Users\NB-UBMI-786\Desktop\VUT\Programs\MyoSeg\pat88\' 

D = dir([path_data '\*.png'])

data=[];
RECT = [112,35,329-112,249-35];

p=1;
for i = 1:length(D) 
    s = split(D(i).name,'_')
    if str2num(s{3})==13
        if str2num(s{4}(1:3))>=2
            img = imread([D(i).folder, '\', D(i).name ]);
            img = imcrop(img, RECT );

            img = histeq(img);
%             img = uint8(((double(img)./255) - 0.0) / (1.9 - 0.0).*255);
            img = uint8((im2double(img).^1.8).*255);

            data(:,:,:,p) = img;
            p = p+1;
        end
    end
end

imshow(img)

path_save = pwd;
fps = 10;

name = 'res_ACDC_time88_5';
save_gif(uint8(data), name, path_save, fps)



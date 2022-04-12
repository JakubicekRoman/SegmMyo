%% load of results mat files from MyoSeg

% path = '/data/rj21/Data/Test_data/Results/Pt04'
path = '/data/rj21/Data/Test_data/Results/Pt04'

D = dir([path '/*2D*.mat'])

img = [];
mask = [];

for i = 1:length(D) 
    load([D(i).folder '/' D(i).name])
    img(:,:,i) = dcm_data;
    mask(:,:,i) = segm_mask;
end

imfuse5(img,mask)
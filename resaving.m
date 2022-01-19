clear all
close all
clc

path_data = ['/data/rj21/Data/Data2/orig'];

path_save = ['/data/rj21/Data/Data2/Resaved'];

D = dir([path_data , '/20*']);
% D(1:2)=[];

for i = 1:length(D)
%     try
    path_save_file = [path_save '/' D(i).name];
    mkdir(path_save_file)

    vel = [128,128,100];
    
    file1 = dir([D(i).folder,'/', D(i).name '/*Reference*.mat']);
    file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T1post.mat']);
    if ~isempty(file2)
        load([D(i).folder,'/', D(i).name '/' file1]);
        niftiwrite(ReferenceMaps.T1map_pre ,[path_save_file '/'  '/Ref_T1post.nii'],'Compressed',true); 
        load([D(i).folder,'/', D(i).name '/' file2],'MyoMask');
        niftiwrite(MyoMask ,[path_save_file '/'  '/Ref_T1post_gt.nii'],'Compressed',true); 
    end

    file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T1pre.mat']);
    if ~isempty(file2)
        load([D(i).folder,'/', D(i).name '/' file1]);
        niftiwrite(ReferenceMaps.T1map_pre ,[path_save_file '/'  '/Ref_T1pre.nii'],'Compressed',true); 
        load([D(i).folder,'/', D(i).name '/' file2],'MyoMask');
        niftiwrite(MyoMask ,[path_save_file '/'  '/Ref_T1pre_gt.nii'],'Compressed',true); 
    end

    file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T2.mat']);
    if ~isempty(file2)
        load([D(i).folder,'/', D(i).name '/' file1]);
        niftiwrite(ReferenceMaps.T1map_pre ,[path_save_file '/'  '/ef_T2.nii'],'Compressed',true); 
        load([D(i).folder,'/', D(i).name '/' file2],'MyoMask');
        niftiwrite(MyoMask ,[path_save_file '/'  '/Ref_T2_gt.nii'],'Compressed',true); 
    end

    file1 = ls([D(i).folder,'/', D(i).name '/*Segmentation.mat']);
    file2 = ls([D(i).folder,'/', D(i).name '/*JointT1T2maps_SA.mat']);

    load([D(i).folder,'/', D(i).name '/' file1],'MyoMask');
    load([D(i).folder,'/', D(i).name '/' file2],'T1_SA');
    load([D(i).folder,'/', D(i).name '/' file2],'T2_SA');

    fileInd =  ls([D(i).folder,'/', D(i).name '/Crop_Indices.mat']);
    load([D(i).folder,'/', D(i).name '/' fileInd]);

    
    Joint_mask = zeros([size(JointT2,1),size(JointT2,2),length(zz)]);
    MyoMask = imresize(MyoMask, [length(xx),length(yy)],'nearest');
    Joint_mask(xx,yy,:) = MyoMask;
    
    T2_SA = T2_SA(:,:,zz);
    T1_SA = T1_SA(:,:,zz);
    vel1 = min(vel,size(T2_SA)); vel1(3)=length(ZZ);
    win = centerCropWindow3d(size(T2_SA),vel1);
    
    JointT2 = imcrop3(T2_SA,win);
    JointT1 = imcrop3(T1_SA,win);
    Joint_mask = imcrop3(Joint_mask,win);
    
    niftiwrite(JointT1 ,[path_save_file '/'  '/Joint_T1.nii'],'Compressed',true); 
    niftiwrite(JointT2 ,[path_save_file '/'  '/Joint_T2.nii'],'Compressed',true); 

    niftiwrite(Joint_mask ,[path_save_file '/'  '/Joint_T1_gt.nii'],'Compressed',true); 
    niftiwrite(Joint_mask ,[path_save_file '/'  '/Joint_T2_gt.nii'],'Compressed',true); 

%     vel(i,:) = size(T2);
% %     catch error
% %         disp( D(i).name)
% %     end
end

% imfuse5(T2,T2_mask)
% imfuse5(JointT2,Joint_mask)




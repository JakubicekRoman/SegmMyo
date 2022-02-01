clear all
close all
clc

% path_data = ['C:/Users/NB-UBMI-786/Desktop/ForRoman_20220114/Database_St' ...
%              'Thomas_annotated/Segmentated_jointT1T2'];
% 
% path_save = ['C:/Users/NB-UBMI-786/Desktop/ForRoman_20220114/'...
%              'Database_StThomas_annotated/Resaved_data_StT'];
  
path_data = ['/data/rj21/Data/Data3/jointT1T2Maps_merged'];

path_save = ['/data/rj21/Data/Data3/Resaved_data_StT_cropped'];

D = dir([path_data , '/20*']);
% D(1:2)=[];

for i = 1:length(D) %[1:30, 32:34]
    %     try
    path_save_file = [path_save '/' D(i).name];
    mkdir(path_save_file)

    % my special crop
    vel = [128,128];
    

    %% for Reference data T1 and T2
    file1 = ls([D(i).folder,'/', D(i).name '/*Reference*.mat']);
    file2=[];
    try file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T1post.mat']); end
    if ~isempty(file2)
        load([file1(1:end-1)]);
        load([file2(1:end-1)],'MyoMask','sl_range', 'Upsampling_factor');
        T = ReferenceMaps.T1map_post(:,:,sl_range);
        MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
        vel1 = max(vel,[size(MyoMask,1),size(MyoMask,2)]); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);
        for m = 1:size(T,3)
            M = ['00000' num2str(m)]; M = M(end-2:end); 
            dicomwrite(uint16(T(:,:,m)) ,[path_save_file '/Ref_T1post_' M '.dcm']); 
            dicomwrite(uint8(MyoMask(:,:,m)) ,[path_save_file '/Ref_T1post_gt_' M '.dcm']); 
        end
    end

    file2=[];
    try file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T1pre.mat']);  end
    if ~isempty(file2)
        load([file1(1:end-1)]);
        load([file2(1:end-1)],'MyoMask','sl_range', 'Upsampling_factor');
        T = ReferenceMaps.T1map_pre(:,:,sl_range);
        MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
        vel1 = max(vel,[size(MyoMask,1),size(MyoMask,2)]); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);
        for m = 1:size(T,3)
            M = ['00000' num2str(m)]; M = M(end-2:end); 
            dicomwrite(uint16(T(:,:,m)) ,[path_save_file '/Ref_T1pre_' M '.dcm']); 
            dicomwrite(uint8(MyoMask(:,:,m)) ,[path_save_file '/Ref_T1pre_gt_' M '.dcm']); 
        end
    end
    
    file2=[];
    try file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T2.mat']); end
    if ~isempty(file2)
        load([file1(1:end-1)]);
        load([file2(1:end-1)],'MyoMask','sl_range', 'Upsampling_factor');
        T = ReferenceMaps.T2map(:,:,sl_range);
        MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
        vel1 = max(vel,[size(MyoMask,1),size(MyoMask,2)]); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);      
        for m = 1:size(T,3)
            M = ['00000' num2str(m)]; M = M(end-2:end); 
            dicomwrite(uint16(T(:,:,m)) ,[path_save_file '/Ref_T2_' M '.dcm']); 
            dicomwrite(uint8(MyoMask(:,:,m)) ,[path_save_file '/Ref_T2_gt_' M '.dcm']); 
        end
    end

    %% for Joint T1/T2 data
    file1 = ls([D(i).folder,'/', D(i).name '/*Segmentation.mat']);
    file2 = ls([D(i).folder,'/', D(i).name '/*JointT1T2maps_SA.mat']);

    load([file1(1:end-1)],'MyoMask','sl_range','Upsampling_factor');
    load([file2(1:end-1)],'T1_SA');
    load([file2(1:end-1)],'T2_SA');
    load([file2(1:end-1)],'Water_1st_SA');
    load([file2(1:end-1)],'Water_4th_SA');

    fileInd =  ls([D(i).folder,'/', D(i).name '/Crop_Indices.mat']);
    load([fileInd(1:end-1)]);

    MyoMask = imresize(MyoMask, 1/Upsampling_factor ,'nearest');
%     MyoMask = imresize(MyoMask, [length(xx),length(yy)],'nearest');
    
    Joint_mask = zeros([size(T2_SA,1),size(T2_SA,2),length(zz)]);
    Joint_mask(xx,yy,:) = MyoMask;

    
    %% cropping
    T2_SA = T2_SA(:,:,zz);
    T1_SA = T1_SA(:,:,zz);
    W1 = Water_1st_SA(:,:,zz);
    W4 = Water_4th_SA(:,:,zz);
    
    T2_SA = uint16( mat2gray(T2_SA)*(2^16) );
    T1_SA = uint16( mat2gray(T1_SA)*(2^16) );
    W1 = uint16( mat2gray(W1)*(2^16) );
    W4 = uint16( mat2gray(W4)*(2^16) );

%     vel1 = min(vel,size(T2_SA)); vel1(3)=length(zz);
%     win = centerCropWindow3d(size(T2_SA),vel1);
%     JointT2 = imcrop3(T2_SA,win);
%     JointT1 = imcrop3(T1_SA,win);
%     Joint_mask = imcrop3(Joint_mask,win);

    vel1 = max(vel,[size(T2_SA,1),size(T2_SA,2)]); 
    JointT2 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), T2_SA);
    JointT1 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), T1_SA);
    W1 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), W1);
    W4 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), W4);
    Joint_mask = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), Joint_mask);

    %% saving
    for m = 1:size(Joint_mask,3)
        M = ['00000' num2str(m)]; M = M(end-2:end); 
        dicomwrite(uint16(JointT1(:,:,m)) ,[path_save_file '/Joint_T1_' M '.dcm']); 
        dicomwrite(uint16(JointT2(:,:,m)) ,[path_save_file '/Joint_T2_' M '.dcm']); 
        dicomwrite(uint16(W1(:,:,m)) ,[path_save_file '/Joint_W1_' M '.dcm']);
        dicomwrite(uint16(W4(:,:,m)) ,[path_save_file '/Joint_W4_' M '.dcm']);
        dicomwrite(uint16(Joint_mask(:,:,m)) ,[path_save_file '/Joint_T1_gt_' M '.dcm']); 
        dicomwrite(uint16(Joint_mask(:,:,m)) ,[path_save_file '/Joint_T2_gt_' M '.dcm']);
        dicomwrite(uint16(Joint_mask(:,:,m)) ,[path_save_file '/Joint_W1_gt_' M '.dcm']);
        dicomwrite(uint16(Joint_mask(:,:,m)) ,[path_save_file '/Joint_W4_gt_' M '.dcm']);
    end

%     vel(i,:) = size(T2);
% %     catch error
% %         disp( D(i).name)
% %     end
end

% imfuse5(T2,T2_mask)
% imfuse5(JointT2,Joint_mask)




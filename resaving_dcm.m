clear all
close all
clc

% path_data = ['C:/Users/NB-UBMI-786/Desktop/ForRoman_20220114/Database_St' ...
%              'Thomas_annotated/Segmentated_jointT1T2'];
% 
% path_save = ['C:/Users/NB-UBMI-786/Desktop/ForRoman_20220114/'...
%              'Database_StThomas_annotated/Resaved_data_StT'];
  
path_data = ['/data/rj21/Data/Data_Joint_StT_Labelled/jointT1T2Maps_merged'];

path_save = ['/data/rj21/Data/Data_Joint_StT_Labelled/dcm_resaved_1mm'];


D = dir([path_data , '/20*']);
% D(1:2)=[];

for i = 1%:length(D) %[1:30, 32:34]
    %     try
    path_save_file = [path_save '/' 'dcm_resaved_1mm_Joint' '/' D(i).name];
    mkdir(path_save_file)

    % my special crop
    vel = [256,256];
    sizePX = 1;

    %% for Reference data T1 and T2
    file1 = ls([D(i).folder,'/', D(i).name '/*Reference*.mat']);
    
    file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T1post.mat']);
    if ~isempty(file2)
        load([D(i).folder,'/', D(i).name '/' file1]);
        load([D(i).folder,'/', D(i).name '/' file2],'MyoMask','sl_range', 'Upsampling_factor');
        T = ReferenceMaps.T1map_post(:,:,sl_range);
        T = imresize(T, Upsampling_factor,'nearest');
        MyoMask = imresize(MyoMask, size(T,[1:2]),'nearest');
        vel1 = max(vel,size(MyoMask,[1:2])); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
%         MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
%         vel1 = max(vel,size(MyoMask,[1:2])); 
%         MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);
        for m = 1:size(T,3)
            M = ['00000' num2str(m)]; M = M(end-2:end); 
            dicomwrite(uint16(T(:,:,m)) ,[path_save_file '/Ref_T1post_' M '.dcm']); 
            dicomwrite(uint8(MyoMask(:,:,m)) ,[path_save_file '/Ref_T1post_gt_' M '.dcm']); 
        end
    end

    file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T1pre.mat']);
    if ~isempty(file2)
        load([D(i).folder,'/', D(i).name '/' file1]);
        load([D(i).folder,'/', D(i).name '/' file2],'MyoMask','sl_range','Upsampling_factor');
        T = ReferenceMaps.T1map_pre(:,:,sl_range);
        T = imresize(T, Upsampling_factor,'nearest');
        MyoMask = imresize(MyoMask, size(T,[1:2]),'nearest');
        vel1 = max(vel,size(MyoMask,[1:2])); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
%         MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
%         vel1 = max(vel,size(MyoMask,[1:2])); 
%         MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);
        for m = 1:size(T,3)
            M = ['00000' num2str(m)]; M = M(end-2:end); 
            dicomwrite(uint16(T(:,:,m)) ,[path_save_file '/Ref_T1pre_' M '.dcm']); 
            dicomwrite(uint8(MyoMask(:,:,m)) ,[path_save_file '/Ref_T1pre_gt_' M '.dcm']); 
        end
    end

    file2 = ls([D(i).folder,'/', D(i).name '/*RefMaps_T2.mat']);
    if ~isempty(file2)
        load([D(i).folder,'/', D(i).name '/' file1]);
        load([D(i).folder,'/', D(i).name '/' file2],'MyoMask','sl_range','Upsampling_factor');
        T = ReferenceMaps.T2map(:,:,sl_range);
        T = imresize(T, Upsampling_factor,'nearest');
        MyoMask = imresize(MyoMask, size(T,[1:2]),'nearest');
        vel1 = max(vel,size(MyoMask,[1:2])); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
%         MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
%         vel1 = max(vel,size(MyoMask,[1:2])); 
%         MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
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

    load([D(i).folder,'/', D(i).name '/' file1],'MyoMask','sl_range','Upsampling_factor');
    load([D(i).folder,'/', D(i).name '/' file2],'T1_SA');
    load([D(i).folder,'/', D(i).name '/' file2],'T2_SA');
    load([D(i).folder,'/', D(i).name '/' file2],'Water_1st_SA');
    load([D(i).folder,'/', D(i).name '/' file2],'Water_4th_SA');

    fileInd =  ls([D(i).folder,'/', D(i).name '/Crop_Indices.mat']);
    load([D(i).folder,'/', D(i).name '/' fileInd]);

%     MyoMask = imresize(MyoMask, 1/Upsampling_factor ,'nearest');
%     MyoMask = imresize(MyoMask, [length(xx),length(yy)],'nearest');

    T2_SA = imresize(T2_SA, Upsampling_factor ,"nearest");
    T1_SA = imresize(T1_SA, Upsampling_factor ,"nearest");
    W_1 = imresize(W_1, Upsampling_factor ,"nearest");
    W_4 = imresize(W_4, Upsampling_factor ,"nearest");
    xx=xx.*2;  yy=yy.*2;
    Joint_mask = zeros([size(T2_SA,1),size(T2_SA,2),length(zz)]);
    Joint_mask(xx(1):xx(end),yy(1):yy(end),:) = MyoMask;

    
    %% cropping
    T2_SA = T2_SA(:,:,zz);
    T1_SA = T1_SA(:,:,zz);
    W_1 = W_1(:,:,zz);
    W_4 = W_4(:,:,zz);

%     vel1 = min(vel,size(T2_SA)); vel1(3)=length(zz);
%     win = centerCropWindow3d(size(T2_SA),vel1);
%     JointT2 = imcrop3(T2_SA,win);
%     JointT1 = imcrop3(T1_SA,win);
%     Joint_mask = imcrop3(Joint_mask,win);

%     vel1 = max(vel,size(T2_SA,[1:2])); 
    vel1 = [256,256];
    JointT2 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), T2_SA);
    JointT1 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), T1_SA);
    W_1 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), W_1);
    W_4 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), W_4);
    Joint_mask = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), Joint_mask);

    %% saving
    for m = 1:size(Joint_mask,3)
        M = ['00000' num2str(m)]; M = M(end-2:end); 
        dicomwrite(uint16(JointT1(:,:,m)) ,[path_save_file '/Joint_T1_' M '.dcm']); 
        dicomwrite(uint16(JointT2(:,:,m)) ,[path_save_file '/Joint_T2_' M '.dcm']); 
        dicomwrite(uint16(W_1(:,:,m)) ,[path_save_file '/Joint_W1_' M '.dcm']); 
        dicomwrite(uint16(W_4(:,:,m)) ,[path_save_file '/Joint_W4_' M '.dcm']);        
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
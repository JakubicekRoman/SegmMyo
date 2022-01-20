clear all
close all
clc

path_data = ['C:\Users\NB-UBMI-786\Desktop\ForRoman_20220114\Database_St' ...
             'Thomas_annotated\Segmentated_jointT1T2'];

path_save = ['C:\Users\NB-UBMI-786\Desktop\ForRoman_20220114\'...
             'Database_StThomas_annotated\Resaved_data_StT'];

D = dir([path_data , '\20*']);
% D(1:2)=[];

for i = [1:30, 32:34] %length(D)
    %     try
    path_save_file = [path_save '\' D(i).name];
    mkdir(path_save_file)

    % my special crop
    vel = [128,128];
    

    %% for Reference data T1 and T2
    file1 = ls([D(i).folder,'\', D(i).name '\*Reference*.mat']);
    file2 = ls([D(i).folder,'\', D(i).name '\*RefMaps_T1post.mat']);
    if ~isempty(file2)
        load([D(i).folder,'\', D(i).name '\' file1]);
        load([D(i).folder,'\', D(i).name '\' file2],'MyoMask','sl_range', 'Upsampling_factor');
        T = ReferenceMaps.T1map_post(:,:,sl_range);
        MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
        vel1 = max(vel,size(MyoMask,[1:2])); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);
        niftiwrite(T ,[path_save_file '\'  '\Ref_T1post.nii'],'Compressed',true); 
        niftiwrite(MyoMask ,[path_save_file '\'  '\Ref_T1post_gt.nii'],'Compressed',true); 
    end

    file2 = ls([D(i).folder,'\', D(i).name '\*RefMaps_T1pre.mat']);
    if ~isempty(file2)
        load([D(i).folder,'\', D(i).name '\' file1]);
        load([D(i).folder,'\', D(i).name '\' file2],'MyoMask','sl_range','Upsampling_factor');
        T = ReferenceMaps.T1map_pre(:,:,sl_range);
        MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
        vel1 = max(vel,size(MyoMask,[1:2])); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);
        niftiwrite(T ,[path_save_file '\'  '\Ref_T1pre.nii'],'Compressed',true); 
        niftiwrite(MyoMask ,[path_save_file '\'  '\Ref_T1pre_gt.nii'],'Compressed',true); 
    end

    file2 = ls([D(i).folder,'\', D(i).name '\*RefMaps_T2.mat']);
    if ~isempty(file2)
        load([D(i).folder,'\', D(i).name '\' file1]);
        load([D(i).folder,'\', D(i).name '\' file2],'MyoMask','sl_range','Upsampling_factor');
        T = ReferenceMaps.T2map(:,:,sl_range);
        MyoMask = imresize(MyoMask, 1/Upsampling_factor,'nearest');
        vel1 = max(vel,size(MyoMask,[1:2])); 
        MyoMask = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), MyoMask);
        T = insertMatrix(zeros(vel1(1),vel1(2),size(T,3)), T);      
        niftiwrite(T,[path_save_file '\'  '\Ref_T2.nii'],'Compressed',true); 
        niftiwrite(MyoMask ,[path_save_file '\'  '\Ref_T2_gt.nii'],'Compressed',true);
    end

    %% for Joint T1/T2 data
    file1 = ls([D(i).folder,'\', D(i).name '\*Segmentation.mat']);
    file2 = ls([D(i).folder,'\', D(i).name '\*JointT1T2maps_SA.mat']);

    load([D(i).folder,'\', D(i).name '\' file1],'MyoMask','sl_range','Upsampling_factor');
    load([D(i).folder,'\', D(i).name '\' file2],'T1_SA');
    load([D(i).folder,'\', D(i).name '\' file2],'T2_SA');

    fileInd =  ls([D(i).folder,'\', D(i).name '\Crop_Indices.mat']);
    load([D(i).folder,'\', D(i).name '\' fileInd]);

    MyoMask = imresize(MyoMask, 1/Upsampling_factor ,'nearest');
%     MyoMask = imresize(MyoMask, [length(xx),length(yy)],'nearest');
    
    Joint_mask = zeros([size(T2_SA,1),size(T2_SA,2),length(zz)]);
    Joint_mask(xx,yy,:) = MyoMask;

    
    %% cropping
    T2_SA = T2_SA(:,:,zz);
    T1_SA = T1_SA(:,:,zz);

%     vel1 = min(vel,size(T2_SA)); vel1(3)=length(zz);
%     win = centerCropWindow3d(size(T2_SA),vel1);
%     JointT2 = imcrop3(T2_SA,win);
%     JointT1 = imcrop3(T1_SA,win);
%     Joint_mask = imcrop3(Joint_mask,win);

    vel1 = max(vel,size(T2_SA,[1:2])); 
    JointT2 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), T2_SA);
    JointT1 = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), T1_SA);
    Joint_mask = insertMatrix(zeros(vel1(1),vel1(2),length(zz)), Joint_mask);

    %% saving
    niftiwrite(Joint_mask ,[path_save_file '\'  '\Joint_T1_gt.nii'],'Compressed',true); 
    niftiwrite(Joint_mask ,[path_save_file '\'  '\Joint_T2_gt.nii'],'Compressed',true); 

    niftiwrite(JointT1 ,[path_save_file '\'  '\Joint_T1.nii'],'Compressed',true); 
    niftiwrite(JointT2 ,[path_save_file '\'  '\Joint_T2.nii'],'Compressed',true); 

%     vel(i,:) = size(T2);
% %     catch error
% %         disp( D(i).name)
% %     end
end

% imfuse5(T2,T2_mask)
% imfuse5(JointT2,Joint_mask)




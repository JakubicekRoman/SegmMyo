
function save_gif(img, name, pathsave, fps)

filename = [pathsave '/' name, '.gif']; % Specify the output file name

nImages = size(img,4);
numCh = size(img,3);

for idx = 1:nImages
    if numCh==3
        [A,map] = rgb2ind(img(:,:,:,idx),256);
    elseif numCh==1
        [A,map] = gray2ind(img(:,:,:,idx),256);
    else
        disp('ERROR: There is be IMG with one or three channel')
        return
    end

    if idx == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1/fps);
    else
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1/fps);
    end
end
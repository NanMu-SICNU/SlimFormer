function sen = getSen(FG, GT)  
    % FG, GT are the binary segmentation and ground truth areas, respectively.  
   % check input
GT = im2double(GT);
if size(GT,3)>1
    GT = GT(:,:,1);
end
FG = im2double(FG);
if size(FG, 3)>1
    FG = FG(:,:,1);
end
if size(GT,1)~=size(FG,1) || size(GT,2)~=size(FG,2)
    FG = imresize(FG, [size(GT,1), size(GT,2)]);
end
FG = ( FG - min(FG(:)) ) ./ ( max(FG(:)) - min(FG(:)) );
GT = ( GT - min(GT(:)) ) ./ ( max(GT(:)) - min(GT(:)) );
    % sensitivity  
    FG = imbinarize(FG);         % 二值化分割图像
    GT  = imbinarize(GT);          % 二值化真值图像
    sen = double(sum(uint8(FG(:) & GT(:)))) / double(sum(uint8(FG(:))));  
end  
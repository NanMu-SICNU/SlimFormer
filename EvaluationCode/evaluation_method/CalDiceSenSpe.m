function [Dice, Sen, Spe] = CalDiceSenSpe(SRC, srcSuffix, GT, gtSuffix)
    
    
    files = dir(fullfile(SRC, strcat('*', srcSuffix)));
if isempty(files)
    error('No saliency maps are found: %s\n', fullfile(SRC, strcat('*', srcSuffix)));
end

    imgDice = zeros(length(files),1);
    imgSen = zeros(length(files),1);
    imgSpe = zeros(length(files),1);
parfor k = 1:length(files)
    srcName = files(k).name;
    srcImg = imread(fullfile(SRC, srcName));
    
    gtName = strrep(srcName, srcSuffix, gtSuffix);
    gtImg = imread(fullfile(GT, gtName));
    
    imgDice(k) = getDice(srcImg, gtImg);
    imgSen(k) = getSen(srcImg, gtImg);
    imgSpe(k) = getSpe(srcImg, gtImg);
end

    Dice = mean(imgDice);
    Sen = mean(imgSen);
    Spe = mean(imgSpe);
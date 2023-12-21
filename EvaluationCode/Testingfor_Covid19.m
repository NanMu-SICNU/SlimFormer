clear all;
close all;
clc;
%%
addpath('./evaluation_method/');
nummodel=9;
gtPath = ('C:/Users/MN/Desktop/PR2021/Covid19_Result/02GT');
gtSuffix = '.png';

pathname = ('C:/Users/MN/Desktop/PR2021/Covid19_Result/');
i=0;

i=i+1;
modelname{i} = '2015FCN';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2015Unet';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2018AttUnet';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2019BASNet';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2019EGNet';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2019PoolNet';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2019UnetAA';
smaptype{i}='.png';
i=i+1;
modelname{i} = '2020InfNet';
smaptype{i}='.png';

i=i+1;
modelname{i} = 'CTnew';
smaptype{i}='.png';
%% Evaluate saliency map
for ii = 1 : nummodel
    salSuffix = smaptype{ii};
    resSalPath = strcat(pathname,modelname{ii},'/');
    
    % compute Precison-recall curve
    [rec(ii,:), pre(ii,:)] = DrawPRCurve(resSalPath, salSuffix, gtPath, gtSuffix, true, true);
    
    % compute ROC curve
    thresholds = [0:1:255]./255;
    [TPR(ii,:), FPR(ii,:)] = CalROCCurve(resSalPath, salSuffix, gtPath, gtSuffix, thresholds);
       
    % compute Dice Sen & Spe
    [Dice(ii),Sen(ii),Spe(ii)] = CalDiceSenSpe(resSalPath, salSuffix, gtPath, gtSuffix);
    
    % compute F-measure curve
    setCurve = true;
    [meanP(ii,:), meanR(ii,:), meanF(ii,:)] = CalMeanFmeasure(resSalPath, salSuffix, gtPath, gtSuffix, setCurve);
    
    % compute MAE
    MAE(ii) = CalMeanMAE(resSalPath, salSuffix, gtPath, gtSuffix);
    
    % compute WF
    Betas = [1];
    WF(ii) = CalMeanWF(resSalPath, salSuffix, gtPath, gtSuffix, Betas);
    
    % compute AUC
    AUC(ii) = CalAUCScore(resSalPath, salSuffix, gtPath, gtSuffix);
    
    % compute Overlap ratio
    setCurve = false;
    overlapRatio(ii) = CalOverlap_Batch(resSalPath, salSuffix, gtPath, gtSuffix, setCurve);
    
end


%% Show results
%% Show results
%r=[0.1 0.4 0.7 0.9 0.5 0.3 0.2 0.4 0.8 0.3 0.8];
%g=[0.5 0.7 0.4 0.1 0.2 0.9 0.2 0.1 0.6 0.4 0.3];
%b=[0.9 0.3 0.2 0.5 0.8 0.1 0.6 0.7 0.4 0.1 0.6];

% show Precison-recall curve
r=[0.1 0.4 0.3 0.7 0.9 0.2 0.5 0.3 0.8];
g=[0.5 0.7 0.9 0.4 0.1 0.2 0.2 0.4 0.3];
b=[0.9 0.3 0.1 0.2 0.5 0.6 0.8 0.1 0.6];
figure, hold on;
for iii = 1 : nummodel-1
plot(rec(iii,:), pre(iii,:), '--','color',[r(iii) g(iii) b(iii)], 'linewidth', 2);
end
plot(rec(iii+1,:), pre(iii+1,:),'r', 'linewidth', 2);
hold off;
grid on;
% lg = columnlegend(3,{ 'NP', 'CA', 'LR', 'PD', 'MR', 'SO', 'BL', 'GP', 'SC', 'SMD', 'MIL','Proposed'},'location', 'southwest','boxon');
h = legend('FCNet','Unet','AttUnet','BASNet', 'EGNet', 'PoolNet', 'UNet++', 'Inf-Net', 'PGPLP',0);

% show ROC curve
figure, hold on;
for iii = 1 : nummodel-1
plot(FPR(iii,:), TPR(iii,:),  '--','color',[r(iii) g(iii) b(iii)], 'linewidth', 2);
end
plot(FPR(iii+1,:), TPR(iii+1,:),'r', 'linewidth', 2);

hold off;
grid on;
% lg = columnlegend(3,{  'NP', 'CA', 'LR', 'PD', 'MR', 'SO', 'BL', 'GP', 'SC', 'SMD', 'MIL','OURS'},'location', 'SouthEast','boxon');
h = legend('FCNet','Unet','AttUnet','BASNet', 'EGNet', 'PoolNet', 'UNet++', 'Inf-Net', 'PGPLP',0);

% show F-measure curve
figure, hold on;
axis([0 250 0.0 1]) 
for iii = 1 : nummodel-1
plot(meanF(iii,:),  '--','color',[r(iii) g(iii) b(iii)], 'linewidth', 2);
end
plot(meanF(iii+1,:),'r', 'linewidth', 2);
hold off;
grid on;
% lg = columnlegend(3,{ 'NP', 'CA', 'LR', 'PD', 'MR', 'SO', 'BL', 'GP', 'SC', 'SMD', 'MIL','OURS'},'location', 'south','boxon');
h = legend('FCNet','Unet','AttUnet','BASNet', 'EGNet', 'PoolNet', 'UNet++', 'Inf-Net', 'PGPLP',0);


% show Dice
fprintf('Dice: %s\n', num2str(Dice));

% show Sen
fprintf('Sen: %s\n', num2str(Sen));

% show Spe
fprintf('Spe: %s\n', num2str(Spe));

% show AUC
fprintf('AUC: %s\n', num2str(AUC));

% show MAE
fprintf('MAE: %s\n', num2str(MAE));

% show WF
fprintf('WF: %s\n', num2str(WF));

% show Overlap ratio
fprintf('OR: %s\n', num2str(overlapRatio));

%% Save results

resPath = 'Covid19resultsnew';
if ~exist(resPath,'file')
    mkdir(resPath);
end

% save Precison-recall curve
PRPath = fullfile(resPath, ['PR.mat']);
save(PRPath, 'rec', 'pre');
fprintf('The precison-recall curve is saved in the file: %s \n', resPath);

% save ROC curve
ROCPath = fullfile(resPath, ['ROC.mat']);
save(ROCPath, 'TPR', 'FPR');
fprintf('The ROC curve is saved in the file: %s \n', resPath);

% save F-measure curve
FmeasurePath = fullfile(resPath, ['FmeasureCurve.mat']);
save(FmeasurePath, 'meanF');
fprintf('The F-measure curve is saved in the file: %s \n', resPath);

% save Dice
DicePath = fullfile(resPath, ['Dice.mat']);
save(DicePath, 'Dice');


% save Sen
SenPath = fullfile(resPath, ['Sen.mat']);
save(SenPath, 'Sen');


% save Spe
SpePath = fullfile(resPath, ['Spe.mat']);
save(SpePath, 'Spe');


% save MAE
MAEPath = fullfile(resPath, ['MAE.mat']);
save(MAEPath, 'MAE');


% save WF
WFPath = fullfile(resPath, ['WF.mat']);
save(WFPath, 'WF');

% save AUC
AUCPath = fullfile(resPath, ['AUC.mat']);
save(AUCPath, 'AUC');

% save Overlap ratio
overlapFixedPath = fullfile(resPath, ['ORFixed.mat']);
save(overlapFixedPath, 'overlapRatio');


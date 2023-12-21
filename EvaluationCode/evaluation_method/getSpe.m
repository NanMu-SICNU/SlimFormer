function spe = getSpe(FG, GT)
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
% specificity
FG = imbinarize(FG);         % ��ֵ���ָ�ͼ��
GT  = imbinarize(GT);          % ��ֵ����ֵͼ��

idx = (GT()==1);

p = length(GT(idx));
n = length(GT(~idx));
N = p+n;

tp = sum(GT(idx)==FG(idx));
tn = sum(GT(~idx)==FG(~idx));
% fp = n-tn;
% fn = p-tp;

fp = p-tp;
fn = n-tn;

acc = (tp+tn)/N;
tp_rate = tp/(tp+fn);
tn_rate = tn/(fp+tn);

sen = tp_rate;  %�����ԣ���������
spe = tn_rate; %�����ԣ���������

%sen = double(sum(uint8(FG(:) & GT(:)))) / double(sum(uint8(FG(:))));

end
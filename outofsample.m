function [res] = outofsample(y,sample)
% �ڶ��׶ξ��߱���out-of-sample performance


[num] = size(sample,1);%��������
%e = ones(n,1);
res = sum((1 + sample)*y)/num;

end


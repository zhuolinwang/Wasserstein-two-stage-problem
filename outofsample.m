function [res] = outofsample(y,sample)
% 第二阶段决策变量out-of-sample performance


[num] = size(sample,1);%样本数量
%e = ones(n,1);
res = sum((1 + sample)*y)/num;

end


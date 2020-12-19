clc
close all
clear all

m2 = 20; %不确定参数维数
n2 = m2; %决策变量y维数
n1 = m2; %决策变量x维数
l = 10; %约束的维数
N = [10]; %样本数
eps = 0.1;
Q = randn(n2,n2)+5;%系数矩阵
q = randn(n2,1)+5; %列向量
W = 3*randn(l,n2)+4;
T = cell(n1);
h = cell(n1);
c = ones(n1,1);

for tt=1:1
t1=clock;
for i = 1:n1
    T{i} = 2*randn(l,m2)+5;
    h{i} = 2*randn(l,1)+10;
end
    
tic
x = binvar(n1,1,'full');
eta = sdpvar(N,1,'full');
y = sdpvar(l,n1,N,'full');
ct = randn(l,n1,N);


T1 = 0;
h1 = 0;
for i = 1:n1
    T1 = T1 + T{i}*x(i);
    h1 = h1 + h{i}*x(i);
end
xi = 4*randn(n1,N(tt))+2;

f = c'*x+sum(sum(sum(ct.*y)))/N(tt);
set = [];
for j =1:N(tt)
    for t=1:l
        set = [set,y(:,:,j)>=0,sum(y(t,:,j))==1];
        set = [set,y(t,:,j)'<=(xi(:,j)-eps).*x];
    end
end

options=sdpsettings('solver','gurobi','verbose',0);
output = optimize(set,f,options);
t2=clock;
tol_m=etime(t2,t1);
% sheet = num2str(N(tt));
% aaa = xlsread('time_2.xlsx',sheet);
% row = size(aaa,1)+1;    
% seq = ['A',num2str(row)];
% xlswrite('time_2.xlsx',tol_m,sheet,seq);
end


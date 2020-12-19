
clc
close all
clear all

K = 20; %不确定参数维数
N2 = K; %决策变量y维数
%N1 = 10; %决策变量x维数
I = [10]; %样本数
J = K;
eps = 0.1;
rho = 0.1;
B = 30;
b = ones(K,1);
ss =10*ones(K,1);


e = ones(K,1);
II = eye(K);

for t=1:1
t1=clock;
xi = 4*randn(K,I(t))+2;
x = sdpvar(K,1,'full'); 
s = sdpvar(I(t),1,'full');
phi = sdpvar(N2+J,I(t),'full');
chi = sdpvar(N2+J,I(t),'full');
theta = sdpvar(1,1,'full');
lambda = sdpvar(1,1,'full');

f = theta + (lambda*eps*eps+sum(s)/I(t))/rho;
set = [x>=0;lambda>=0;s>=0;e'*x<=B];
Tx = [-diag(b) diag(ss)]';
hx = [x'*diag(b) -x'*diag(ss)]';
W = [eye(2*K)]';
col = K+size(Tx,1);
M = cell(I);
P = cell(I);
N = cell(I);
for i= 1:I(t)
   M{i} = [lambda*eye(K) -0.5*Tx' -lambda*xi(:,i)
     -0.5*Tx W*diag(phi(:,i))*W' 0.5*(W*phi(:,i)-hx)
     -lambda*xi(:,i)' 0.5*(W*phi(:,i)-hx)' s(i)+theta-sum(phi(:,i)+chi(:,i))+lambda*xi(:,i)'*xi(:,i)];
   [m,n]=size(M{i});
   P{i} = sdpvar(m,m); 
   N{i} = sdpvar(m,n,'full');
   set = [set;P{i}>=0,P{i}+N{i}==M{i},N{i}>=0];
end
%options=sdpsettings('solver','cplex','verbose',0);
options=sdpsettings('solver','mosek','verbose',0);
output = optimize(set,f,options);
t2=clock;
tol_m=etime(t2,t1);
% sheet = num2str(I(t));
% aaa = xlsread('time_1.xlsx',sheet);
% row = size(aaa,1)+1;    
% seq = ['A',num2str(row)];
% xlswrite('time_1.xlsx',tol_m,sheet,seq);
end


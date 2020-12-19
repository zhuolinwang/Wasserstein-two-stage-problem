%% uncertainty in objective: test the impact of the sample size N
clc
clear all
close all

%% 估计第一阶段的回报（r1）
DJA_5 = load('DJA_5');
DJA_5 = DJA_5.DJA_5;

DJI_5 = load('DJI_5');
DJI_5 = DJI_5.DJI_5;

DJT_5 = load('DJT_5');
DJT_5 = DJT_5.DJT_5;

DJU_5 = load('DJU_5');
DJU_5 = DJU_5.DJU_5;


%% 第二阶段关于回报的样本(sample for second-stage return)
DJA_2 = load('DJA_2');
DJA_2 = DJA_2.DJA_2;

DJI_2 = load('DJI_2');
DJI_2 = DJI_2.DJI_2;

DJT_2 = load('DJT_2');
DJT_2 = DJT_2.DJT_2;

DJU_2 = load('DJU_2');
DJU_2 = DJU_2.DJU_2;

DJA_1 = load('DJA_1');
DJA_1 = DJA_1.DJA_2019;

DJI_1 = load('DJI_1');
DJI_1 = DJI_1.DJI_2019;

DJT_1 = load('DJT_1');
DJT_1 = DJT_1.DJT_2019;

DJU_1 = load('DJU_1');
DJU_1 = DJU_1.DJU_2019;

DJA_2010 = load('DJA_2010');
DJA_2010 = DJA_2010.DJA_2010;

DJI_2010 = load('DJI_2010');
DJI_2010 = DJI_2010.DJI_2010;

DJT_2010 = load('DJT_2010');
DJT_2010 = DJT_2010.DJT_2010;

DJU_2010 = load('DJU_2010');
DJU_2010 = DJU_2010.DJU_2010;


sample_1 = [DJA_2(:,1),DJI_2(:,1),DJT_2(:,1),DJU_2(:,1)];%2017年的样本，每一行作为一个样本
sample_2 = [DJA_2(:,2),DJI_2(:,2),DJT_2(:,2),DJU_2(:,2)];%2018年的样本
sample_3 = [DJA_1,DJI_1,DJT_1,DJU_1].*10;
sample = [sample_1(103:end,:);sample_2;].*10;%两年的样本


r1 = mean(sample_1(1:102,:),1)';
r1 = r1+[mean(DJA_5);mean(DJI_5);mean(DJT_5);mean(DJU_5)];
r1 = r1.*10;
r = 0;
sam_num = 20:5:200;


%% 测试样本数量的影响
tic

[~,n] = size(sample);
c = -ones(n,1) - r1;
A = diag(-c);
theta = 0.01;
eps = 0.02;
num = length(sam_num);

for time = 1:1
    sample = sample(randperm(400),:);
    test = sample(201:end,:);
    
   
    xx = zeros(n,num);
    yy = zeros(n,num);
    ff = zeros(n,1);
    res = zeros(num,1);

    for i=1:num
        train = sample(1:sam_num(i),:); 
        x = sdpvar(n,1); %第一阶段决策变量
        lambda = sdpvar(1,1);
        s = sdpvar(sam_num(i),1);
        y = sdpvar(n,1); %第二阶段决策变量
        delta_p = sdpvar(n,1);
        delta_n = sdpvar(n,1);

        f = c'*x + lambda*eps + sum(s)/sam_num(i); %目标函数
        set_1 = [y >= 0;y == A*x+delta_n-theta*delta_p;norm(y) <= lambda;delta_n <= delta_p;delta_n >= delta_p;delta_p >= 0;
                 theta*sum(delta_p) == sum(A*x-y);(1+theta)*0.5*(delta_p-delta_n) <= A*x;(1+theta)*0.5*(delta_p+delta_n)<= sum(A*x)-A*x;];
        set_2 = [(-1-train)*y <= s; x >= 0; sum(x) == 1;];

        set = [set_1;set_2];
        options=sdpsettings('solver','Mosek','verbose',0); 
        optimize(set,f,options);
        xx(:,i) = double(x);
        yy(:,i) = double(y);
        ff(i) = double(f);
        res(i) = outofsample(yy(:,i),test);
    end
    r = r + res;
end
toc


    




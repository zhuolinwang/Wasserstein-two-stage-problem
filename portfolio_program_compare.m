%% compare

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



r_m = 0;
r_w = 0;
r_s = 0;

mu = rand(1,4);

inv_mu = (2)./mu;
sigma = diag(inv_mu);
sample = sample(randperm(400),:)+ 0.1*(mvnrnd(mu',sigma,400));

%% 优化求解
tic
for time = 1:100
    sample = sample(randperm(400),:);
    train = sample(1:50,:);
    [m,n] = size(train);
    test = sample(301:end,:);
     
    [m,n] = size(train);
    mu = mean(train,1)';
    sigma = cov(train);
    var = std(train)';%标准差
    gamma_0 = 10;
    gamma_q = 1;
  
    c = -ones(n,1) - r1;
    A = diag(-c);
    theta = 0.01;
    eps = 0.1;

    num = length(eps);
    xx = zeros(n,num);
    yy = zeros(n,num);
    ff = zeros(n,1);
    
    xx1 = zeros(n,num);
    yy1 = zeros(n,num); 
    yys = zeros(n,num);
    ff1 = zeros(n,1);
    res = zeros(num,1);
    res1 = zeros(num,1);
    ress = zeros(num,1);

    for i=1:num
        x = sdpvar(n,1); %第一阶段决策变量
        z = sdpvar(1,1);
        Z = sdpvar(n,n);
        w_1 = sdpvar(n,1);
        w_2 = sdpvar(n,1);
        y = sdpvar(n,1); %第二阶段决策变量
        delta_p = sdpvar(n,1);
        delta_n = sdpvar(n,1);
        
       % Moment
        f = c'*x +z+w_1'*mu+w_2'*gamma_q*var + trace(Z*(gamma_0*sigma+mu*mu')); %目标函数
        set_1 = [y >= 0;y == A*x+delta_n-theta*delta_p;delta_n <= delta_p;delta_n >= delta_p;delta_p >= 0;
                 theta*sum(delta_p) == sum(A*x-y);(1+theta)*0.5*(delta_p-delta_n) <= A*x;(1+theta)*0.5*(delta_p+delta_n)<= sum(A*x)-A*x;];
        set_2 = [x >= 0; sum(x) == 1;];
        
        set_3 = [[Z,0.5*(w_1-y);0.5*(w_1-y)',z]>=0,w_2>=0;Z>=0;];

        set = [set_1;set_2;set_3];
        options = sdpsettings('solver','Mosek','verbose',0); 
        optimize(set,f,options);
        xx(:,i) = double(x);
        yy(:,i) = double(y);
        ff(i) = double(f);
        res(i) = outofsample(yy(:,i),test);
        
        % Wasserstein
        lambda = sdpvar(1,1);
        y_w = sdpvar(n,1);
        x_w = sdpvar(n,1);
        s = sdpvar(m,1);
        f_1 = c'*x_w + lambda*eps + sum(s)/m; %目标函数
        set1 = [y_w >= 0;y_w == A*x_w+delta_n-theta*delta_p;norm(y_w) <= lambda;delta_n <= delta_p;delta_n >= delta_p;delta_p >= 0;
                 theta*sum(delta_p) == sum(A*x_w-y_w);(1+theta)*0.5*(delta_p-delta_n) <= A*x_w;(1+theta)*0.5*(delta_p+delta_n)<= sum(A*x_w)-A*x_w;];
        set2 = [(-1-train)*y_w <= s; x_w >= 0; sum(x_w) == 1;];

        sett = [set1;set2];
        options = sdpsettings('solver','Mosek','verbose',0); 
        optimize(sett,f_1,options);
        xx1(:,i) = double(x_w);
        yy1(:,i) = double(y_w);
        ff1(i) = double(f_1);
        res1(i) = outofsample(yy1(:,i),test);
        


    
    end
        

   
    r_m = r_m + res; %optimal function value for moment-based method
    r_w = r_w+res1;%optimal function value for Wasserstein method

end
toc



    





    







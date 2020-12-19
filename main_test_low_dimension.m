
%% uncertainty in constraints - test in R^4 dimension
clc
clear all
close all

%% ������ʼ��

c = [2;3];
q = [7;12];
u = 100;
N = 500;
%N = [10,20,30,50,100,200,300,500,1000]; %��������
%eps = 1/sqrt(N);%Wasserstein ball �뾶
%eps = 0.01;

B = eye(2,2);
A0 = [2 3;6,3.4]; %����A��ȡֵ
A1 = [1,0;0,0];
A2 = [0,0;0,1];
A3 = [0,0;0,0];
A4 = [0,0;0,0];

b0 = [180;162]; %����b��ȡֵ
b1 = [0;0];
b2 = [0;0];
b3 = [1;0];
b4 = [0;1];

eps = 0.01:0.05:1;
%eps = 0.1;
len = length(eps);
time = 1; %�ظ�����
n1 = length(c); %��һ�׶α���ά��


cost1 = zeros(len,time); %��ͬeps�µ�һ�׶ε�cost
OFV = zeros(len,time); %����������ֵ
xx = zeros(n1,len,time);

cost1_u = zeros(len,time); %��ͬeps�µ�һ�׶ε�cost
OFV_u = zeros(len,time); %����������ֵ
xx_u = zeros(n1,len,time);

cost_p = zeros(len,time); %��ͬeps�µ�һ�׶ε�cost
OFV_p = zeros(len,time); %����������ֵ
xxp = zeros(n1,len,time);
e_num = zeros(time,len);
kk = zeros(time,len);

 
for t=1:time
    
    mu = [0;0;0;0];
    sig = diag([9,12,0.21,0.16]);   
    sample = mvnrnd(mu,sig,N); %������������ ÿһ����һ������
    [~,n2] = size(sample); %�ڶ��׶�����ά��
    newsample = sample;
    %% �Ż���� Algorithm1
    for i=1:len
        index = 1:N;
        E = [0;0]; %��ʼ����ļ��ϣ�ÿһ����һ������
        k = 0;
        upper = +1000000000;
        lower = -1000000000;
        ite = 0;
        up = [];lo = [];
        time1=clock;
        while(abs((upper - lower)/lower) > 5*10^-5)
            old = abs((upper - lower)/lower);
            k = k+1;
            [dim_e,enum] = size(E);
            x = sdpvar(n1,1);
            lambda = sdpvar(1,1);
            %N = size(newsample,1);
            ss = sdpvar(N,1);
            C = [(b1 - A1*x)';(b2 - A2*x)';(b3 - A3*x)';(b4 - A4*x)'];
            P_C = C*E; %�任֮��ļ���

            %% master
            f = c'*x+lambda*eps(i)+sum(ss)/N;
            set_1 = [sum(x) <= u;x >= 0];
            S = repmat(ss,1,enum);
            temp = (b0-A0*x)'*E;
            set_2 = sample*(C*E) + repmat(temp,N,1) <= S;
            for j=1:enum
                set_2 = [set_2;lambda >= norm(P_C(:,j))];
            end
            set = [set_1;set_2];
            options=sdpsettings('solver','Mosek','verbose',0); 
            optimize(set,f,options);

            %% subproblem
            x_1 = double(x);
            C_1 = double(C); %�任����
            temp_1 = double(temp);
            lambda = double(lambda);
            maxs = double(ss);

            cc = b0 - A0*x_1;
            [E,flag_1,newsample,index,su] = linear_p(C_1,cc,newsample,E,maxs,index,q);

            P = -C_1'*C_1; ub = q; lb = zeros(length(q),1);
            P = P./(-min(eig(P))*2);
            rho =  1.2; alpha = 0.8;

            [y, ~] = quadprog_admm(P,lb, ub, rho, alpha);
            lambda_f = norm(C_1*y);

            if ~ismember(y', E', 'rows')
                E = [E,y]; %�������������ļ������
                E = unique(E','rows');
                E = E';                           
            elseif upper == 1000000000 %��һ�θ���
                upper = c'*x_1+lambda_f*eps(i)+su/N;
            end 
            upper = min(upper,c'*x_1+lambda_f*eps(i)+su/N);
            lower = double(f);
            up(k) = upper;
            lo(k) = lower;

            if abs(old - abs((upper - lower)/lower))<=10^-5
                ite = ite + 1;
            else
                ite = 0;
            end
            if ite >= 5
                break;
            end
        end
        time2=clock;
        tol=etime(time2,time1); 
        [~,enum] = size(E);
        maxs = round(maxs,-4);
        cost1(i,t) = c'*x_1; %��ͬeps�µ�һ�׶ε�cost
        OFV(i,t) = double(f); %����������ֵ
        xx(:,i,t) = x_1;
        OFV_u(i,t) = upper; 
        e_num(t,i) = enum; %record the extreme points number and interation number 
        kk(t,i) = k;
    end

        
end
%% pic

figure
plot(eps,OFV_u,'-d','MarkerSize',6,'linewidth',1.5);

title('O.F.V')
hold on
plot(eps,OFV,'-+','MarkerSize',9,'linewidth',1.5);



hh=legend('Upper bound','Lower bound');
xlabel('\epsilon_N','FontSize',12);
grid on




len_it = length(lo);
figure
plot([1:len_it],up,'-d','MarkerSize',6,'linewidth',1.5);

hold on
plot([1:len_it],lo,'-+','MarkerSize',9,'linewidth',1.5);


h=legend('Upper bound','Lower bound');
xlabel('Iteration number','FontSize',12);
grid on

%% The number of the Extreme points and the iteration in one experiment
disp(['The number of the extreme points generated in this Algorithm is ��N = ',num2str(N),' esp = ',num2str(eps(3)),')']);
disp(e_num(3));
disp(['The number of the extreme points generated in this Algorithm is ��N =: ',num2str(N),' esp = ',num2str(eps(3)),')']);
disp(kk(3));



%% uncertainty in constraints：Test in high dimension convergence
    clc
    clear all  
    close all

    %% 参数初始化
    c = [2;3;1;4;5;2;4;3;4;2;5;4;4;2;6;2;4;3;1;2];
    lenx = length(c);
    q = [7;9;4;6;8;5;6;8;10;7;12;10;6;7;9;5;11;10;5;8];
    u = 1000;
  %  N = [10,20,30,50,100,200,300,1000]; %sample size
    N = 200;


    B = eye(lenx,lenx);
    A0 =load('A0.mat'); %矩阵A的取值
    A0 = A0.A0;
    A = cell(lenx*2,1);
    for i = 1:lenx
         data = zeros(lenx,lenx);
         data(i,i)=1;
         A{i}=data;
    end

    for i = 1:lenx
         data = zeros(lenx,lenx);
         A{i+lenx}=data;
    end


    b0 = [180;162;150;152;150;182;142;130;152;170;190;142;150;202;160;170;175;140;150;160;]*2; %向量b的取值
    b = cell(lenx*2,1);
    for i = 1:lenx
         data = zeros(lenx,1);
         data(i)=1;
         b{i+lenx} = data;
    end

    for i = 1:lenx
         data = zeros(lenx,1);
         b{i} = data;
    end

    eps = 0.01:0.05:1;
    len = length(eps);
    time = 1; %重复试验
    n1 = length(c); %第一阶段变量维度
    k = 1;
    kk = zeros(time,len);


    cost1 = zeros(len,time); %不同eps下第一阶段的cost
    OFV = zeros(len,time); %函数的最优值
    xx = zeros(n1,len,time);

    cost1_u = zeros(len,time); %不同eps下第一阶段的cost
    OFV_u = zeros(len,time); %函数的最优值
    xx_u = zeros(n1,len,time);

    e_num = zeros(time,len);


    %%
 tic
    for t=1:time 
            mu = zeros(40,1)';
            sig1 = rand(20,1)+5;
            sig2 = rand(20,1);
            sig = diag([sig1',sig2']);
            sample = mvnrnd(mu,sig,N); %生成样本矩阵 每一行是一个样本
            [~,n2] = size(sample); %第二阶段样本维度        
        %% 优化求解 Algorithm1
        for i=1:len      
            flag_1 = 1; flag_2 = 1;
            newsample = sample;
            index = 1:N;
            E = zeros(20,1); %初始极点的集合，每一列是一个极点
            %E = [7;9;4;6;8;5;6;8;10;7;12;10;6;7;9;5;11;10;5;8];
            k = 0; 
            upper = +1000000000;
            lower = -1000000000;
            up = [];lo = [];
            ite = 0;
            tic
            time1=clock;
            while(abs((upper - lower)/lower) > 5*10^-4)
                old = abs((upper - lower)/lower);  
                [dim_e,enum] = size(E);
                x = sdpvar(n1,1);
                lambda = sdpvar(1,1);
                ss = sdpvar(N,1);
                C =[];
                for ii=1:40
                    C = [C;(b{ii}-A{ii}*x)'];
                end        
                P_C = C*E; %变换之后的极点

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
                optimize(set,f,options); %求解得s以及p
                k = k+1;%OFV_k = [OFV_k,double(f)];

                %% subproblem
                x_1 = double(x);
                C_1 = double(C); %变换矩阵
                temp_1 = double(temp);
                maxs = double(ss);
                lambda = double(lambda);
                cc = b0 - A0*x_1;
                [E,flag_1,newsample,index,su] = linear_p(C_1,cc,newsample,E,maxs,index,q);
                P = -C_1'*C_1; ub = q; lb = zeros(length(q),1);
                P = P./(-min(eig(P))*2);
                rho =  1.2; alpha = 0.8;
                [y, ~] = quadprog_admm(P,lb, ub, rho, alpha);
                lambda_f = norm(C_1*y);

                if ~ismember(y', E', 'rows')
                    E = [E,y]; %将不符合条件的极点加入
                    E = unique(E','rows');
                    E = E';                                    
                elseif upper == 1000000000
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
                if ite >= 3
                    break;
                end
                

            end
            toc
            time2 = clock;
            tol=etime(time2,time1); 
            cost1(i,t) = c'*x_1; %不同eps下第一阶段的cost
            maxs = round(maxs,-4);
            OFV(i,t) = lower; %函数的最优值下界
            OFV_u(i,t) = upper; %函数的最优值上界
            xx(:,i,t) = x_1;
            e_num(t,i) = enum; %record the extreme points number and interation number 
            kk(t,i) = k;
        end
        
       % ofvk{t} = OFV_k;
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
disp(['The number of the extreme points generated in this Algorithm is （N = ',num2str(N),' esp = ',num2str(eps(3)),')']);
disp(e_num(3));
disp(['The number of the extreme points generated in this Algorithm is （N =: ',num2str(N),' esp = ',num2str(eps(3)),')']);
disp(kk(3));


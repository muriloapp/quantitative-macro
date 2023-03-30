%% Problem Set 1 - Macroeconomics III

%% Question 1 - Bisection
f=@(x) exp(x) - exp(2.2087);  %Function f(x)=e^x - e^2.2087
a=0;          % Lower bound of the the interval x in[a,b]
b=4;          % Upper bound of the the interval x in[a,b]
root= bisection(f,a,b);

disp('Root of the function')

root;

%% Question 2 - Solving Nonlinear Equations - Newton's Method.

%% Item a. 

clear all
c=1; 
fplot(@(x) x^(-5) - x^(-3) -c,[0.6 10]) %f = @(x)x^(-5) - x^(-3) - c;
title('Plot question 2.a')

% Maximum number of iterations
maxit = 1000; 
%Tolerance value
crit=1e-3; %
%parameters
param = [-5;-3;1]; 
%seed
x0=0.6; 

%Solving by newton's method
sol = newton('f2a',x0,param,crit,maxit); 
sprintf('The solution of the problem 2.a is %g',sol)

%% Item b.

c1 = linspace(1,10,10); %equidistant grid for c containing 10 nodes between 1 and 10,
maxit=1000; % Maximum number of iterations
x0=0.6;

for i = 1:10
[sol1(i)]=newton('f2a', x0, [-5;-3;c1(i)], 1e-3, maxit);
end
%With this we reach the row vector with the solutions of the problem

%% Item c.

c2 = linspace(1,10,1000); %equidistant grid for c containing 1000 nodes between 1 and 10,
x0 = 0.6;
maxit = 1000;

for i = 1:1000
[sol2(i)]=newton('f2a', x0, [-5;-3;c2(i)], 1e-3, maxit);
end

% Use the buil-in function to cubic interpolation
x = c2;   
v =sol2;        
xq = x; 

vq = spline(x,v,xq); % Linear Interpolation

% Show the picture
plot(x,v,'o',xq,vq,':.');
xlim([1 10]);
title('Cubic Spline Interpolation');

%% Item d.

%Relabel solution from b. (Recall that solution from b is sol1)
sol3 = sol1;

% Use the buil-in function to cubic interpolation
x = sol3; % This defines the values of x to be the solution find in Question_2_b
v = c1; % Correspondents values of c      
xq = linspace(0.6,1,1000); %This is the query points

vq = spline(x,v,xq); 

% Show the picture
plot(x,v,'o',xq,vq,':.');
xlim([0.6 1]);
title('Cubic Spline Interpolation of c(x)');

%% Item e.

% Now I'll find the solution to: 0 = c(x) + x

%Recall that we have the following variables in our workspace
%maxit = 1000
%crit = 1.000e-03
%x0 = 0.6

%Solving by newton's method
sol4 = newton('f2e',x0, [-5;-3], crit, maxit);
sprintf('The solution of the problem 2.e is %g',sol4)

%% Question 3 - Approximation Methods: Finite Element Methods.

%% Item a.

%Generate an aproximation to h(x) on the domain x in [-2,2]
clear all
%n=5 equally spaced nodes
x=-2:1:2; 

%Assigning values 
for i=1:5 
    if x(i)<=0;
        v(i)=(x(i)+0.5)^2;
    else 
        v(i)=(x(i)-0.5)^2;
    end
end

%Coordinates of the query points over the fine grid, with interval 0.0001
xq=-2:0.0001:2; 
%Cubic interpolation
vq=spline(x,v,xq);

%Plot
hold on
plot(x,v,'o',xq,vq,':.')
fplot(@(x) (x + 0.5)^2,[-2 0],'k')
fplot(@(x) (x - 0.5)^2,[0 2],'k')
hold off
legend('Nodes','Cubic Spline Interpolation with 5 nodes','Real Function')

%Size of the grid from interpolation
h=size(xq);

%Computing squared errors
for i=1:h(1,2) 
    if xq(i)<0;
        err_a(i)=((xq(i)+0.5)^2 - vq(i))^2;
    else 
        err_a(i)=((xq(i)-0.5)^2 - vq(i))^2;
    end
end

%Mean squared error
MSEa = mean(err_a)
%Root mean squared error
RMSEa = MSEa^0.5;
RMSEa %Show Mean squared error

%% Item b.

%Generate an aproximation to h(x) onde the domain x in [-2,2]
%n=10 equally spaced nodes
x_b = linspace(-2,2,10); 

%Assigning values
for i=1:10
    if x_b(i)>=0;
        v_b(i)=(x_b(i)-0.5)^2;
    else
        v_b(i)=(x_b(i)+0.5)^2;
    end
end

%Coordinates of the query points over the fine grid, with interval 0.0001
xq_b=-2:0.0001:2; 
%Cubic interpolation
vq_b=spline(x_b,v_b,xq_b);

%Plot
hold on
plot(xq_b,vq_b,'b:.')
plot(xq,vq,':.')
fplot(@(x) (x + 0.5)^2,[-2 0],'k')
fplot(@(x) (x - 0.5)^2,[0 2],'k')
hold off
legend('Cubic Spline Interpolation with 10 nodes', 'Cubic Spline Interpolation with 5 nodes', 'Real function')

%Size of the grid from interpolation
h=size(xq_b);

%Computing squared errors
for i=1:h(1,2) 
    if xq_b(i)<0;
        err_b(i)=((xq_b(i)+0.5)^2 - vq_b(i))^2;
    else 
        err_b(i)=((xq_b(i)-0.5)^2 - vq_b(i))^2;
    end
end

%Mean squared error
MSEb = mean(err_b)
%Root mean squared error
RMSEb = MSEb^0.5;
RMSEb %Show Mean squared error

%% Question 4 - Estimating pi with Monte Carlo procedure

%% Item a.

clear all
np1=100; %Total number of points we want to generate, in this case 100.

for i=1:np1 
    x1=rand(i,1); %'i' row, 1 column vector of random x values will be generated with each iteration of the for loop
    y1=rand(i,1); %same thing for y
    
end

check_in1=(x1-0.5).^2 + (y1-0.5).^2 < 0.5^2; %I am considering here a circle with r=0.5 and centered in (0.5,0.5).
pi_est1=sum(check_in1)*4/np1; %calculation of pi. Sum up the number of points within the circle, multiply by 4, and then divide by the number of points generated


fprintf('Monte Carlo (Circle Area) Results:\n ');

%% Item b.

theta=0:0.01:2*pi; %set of angles for the circle
circlex=0.5+0.5*cos(theta); %function to create the x values of the circle. 
%circle is centered at (0.5,0.5) and has a radius of 0.5
circley=0.5+0.5*sin(theta); %function to create y values of circle



plot(x1,y1,'.'); hold on; %populate the square with random points
plot(circlex,circley,'r','linewidth',1.5); hold off; %plot the circle inside of the 
%square. each side of the square should be tangential to the edge of the circle
axis equal
title('Using Circle Area to Estimate Pi, number of points = 100');


%% Item c.

np=5000:5000:100000;
%check_in=zeros(20,1);
%pi_est = zeros(20,1);

 %Replicating the algorithm 20 times, using diferent amounts of random draws (n = 5000; 10000; 15000; :::; 100000).
for j=1:20
    x=rand(np(j),1); %'i' row, 1 column vector of random x values will be generated with each iteration 
    y=rand(np(j),1); 
    
check_in=(x-0.5).^2 + (y-0.5).^2 < 0.5^2; %I am considering here a circle with r=0.5 and centered in (0.5,0.5).
pi_est(j)=sum(check_in)*4/np(j); %sum up the number of points within the circle, multiply by 4, and then divide by the number of points generated during THAT particular iteration
    
end


plot(np, pi_est, ':.');xlabel('nº of points');ylabel('Estimated pi'); grid on;
axis([5000 100000 3.11 3.18]);

%% Question 5 - Two Period Model

%% Item c

clear all

%Parameters
A0  =  1;
A1  =  1;   
alpha  =  2/3;
sigma  =  2;
beta =  0.98^25 ; %subjective discount rate of 2% per year, but a model period is 25 years
gamma = 1;
r = (1+0.05)^25-1; %r = 5% per year, but a model period is 25 years

% Vector of parameters
P = [A0, A1, alpha, sigma, beta, gamma, r];

options=optimset('Display','iter', 'MaxFunEvals', 20000, 'MaxIter', 20000, 'TolFun', 1e-10, 'TolX', 1e-10);

% Initial guess
w0 = [1	    1];
%    [w0   w1]

%Solving the FOC to find w0, w1
[w, f_val] = fsolve('foc5',w0, options, P);
%Model solution
sol_c=model_solution(P,w)
sol_c = array2table(sol_c, 'VariableNames', {'w0', 'w1', 'L0','L1','Y0','Y1','l0','l1','c0','c1','a1'});
sol_c


%% Item d

%Changin the parameter
A0  =  0.9;
P = [A0, A1, alpha, sigma, beta, gamma, r]; % Vector of parameters
options=optimset('Display','iter', 'MaxFunEvals', 20000, 'MaxIter', 20000, 'TolFun', 1e-10, 'TolX', 1e-10);

%Solving the FOC to find w0, w1
[w, f_val] = fsolve('foc5',w0, options, P);

%Model Solution
sol_d=model_solution(P,w)
sol_d = array2table(sol_d, 'VariableNames', {'w0', 'w1', 'L0','L1','Y0','Y1','l0','l1','c0','c1','a1'});
sol_d

%% Item e

%Changing parameters
A0  =  1;
A1  =  0.9;
P = [A0, A1, alpha, sigma, beta, gamma, r]; % Vector of parameters
options=optimset('Display','iter', 'MaxFunEvals', 20000, 'MaxIter', 20000, 'TolFun', 1e-10, 'TolX', 1e-10);

%Solving the FOC to find w0, w1
[w, f_val] = fsolve('foc5',w0, options, P);

%Model Solution
sol_e=model_solution(P,w)
sol_e = array2table(sol_e, 'VariableNames', {'w0', 'w1', 'L0','L1','Y0','Y1','l0','l1','c0','c1','a1'});
sol_e


%% Item f

%Changing parameter sigma
sigma  =  1.5;

%Redo iten (c)
A0  =  1;
A1  =  1; 
% Vector of parameters
P = [A0, A1, alpha, sigma, beta, gamma, r];
options=optimset('Display','iter', 'MaxFunEvals', 20000, 'MaxIter', 20000, 'TolFun', 1e-10, 'TolX', 1e-10);
%Solving the FOC to find w0, w1
[w, f_val] = fsolve('foc5',w0, options, P);
%Model solution
sol_f_c=model_solution(P,w)
sol_f_c = array2table(sol_f_c, 'VariableNames', {'w0', 'w1', 'L0','L1','Y0','Y1','l0','l1','c0','c1','a1'});
sol_f_c



%Redo iten (d)
A0  =  0.9;
A1  =  1; 
% Vector of parameters
P = [A0, A1, alpha, sigma, beta, gamma, r];
options=optimset('Display','iter', 'MaxFunEvals', 20000, 'MaxIter', 20000, 'TolFun', 1e-10, 'TolX', 1e-10);
%Solving the FOC to find w0, w1
[w, f_val] = fsolve('foc5',w0, options, P);
%Model solution
sol_f_d=model_solution(P,w)
sol_f_d = array2table(sol_f_d, 'VariableNames', {'w0', 'w1', 'L0','L1','Y0','Y1','l0','l1','c0','c1','a1'});
sol_f_d


%Redo iten (e)
A0  =  1;
A1  =  0.9; 
% Vector of parameters
P = [A0, A1, alpha, sigma, beta, gamma, r];
options=optimset('Display','iter', 'MaxFunEvals', 20000, 'MaxIter', 20000, 'TolFun', 1e-10, 'TolX', 1e-10);
%Solving the FOC to find w0, w1
[w, f_val] = fsolve('foc5',w0, options, P);
%Model solution
sol_f_e=model_solution(P,w)
sol_f_e = array2table(sol_f_e, 'VariableNames', {'w0', 'w1', 'L0','L1','Y0','Y1','l0','l1','c0','c1','a1'});
sol_f_e

%% Question 6 - Growth Model

%% Items d,e
clear all

% Parameters (considering variables c and k per effective labor)
beta=0.98;
delta=0.08;
gamma=0.015;
alpha=0.4;
theta=2;
tauc=0.15;
tauh=0.25;
tauk=0.15;
eta=0;  
T=100;      

%Initial values and Variables in the steady state
Upsilon = (((1+gamma)-beta*(1-delta))/(beta*alpha*(1-tauk)))^(1/(1-alpha));
Delta = delta +  gamma + eta + eta*gamma;
%Steady state k
kss = ((1-alpha)*(1-tauh)*(Upsilon.^(-alpha)))/(theta*(Upsilon.^(1-alpha))*(1-tauh.*(1-alpha) - tauk*alpha) - theta*Delta + (1-alpha).*(1-tauh).*(Upsilon.^(1-alpha)));
k0 = 0.8*kss;
%Steady state h
hss = Upsilon*kss;
h0 = 0.8*hss;
%Steady state c
css = (((kss^(alpha))*(hss^(1-alpha))*(1-tauh*(1-alpha) - tauk*alpha)) - kss*Delta)/(1+tauc);
c0 = 0.8*css;
%Steady state g
gss = tauc.*css + tauh.*(1-alpha).*(kss.^(alpha)).*(hss.^(1-alpha)) + tauk.*(alpha).*(kss.^(alpha)).*(hss.^(1-alpha));
g0 = 0.8*gss;

for j=1:T;
xk(j)=k0*(1-j/T)+j/T*kss; 
xh(j)=h0*(1-j/T)+j/T*hss;
xg(j)=g0*(1-j/T)+j/T*gss;
end

x0=[xk',xh',xg'];
param=[beta,delta,gamma,alpha,theta,tauc,tauh,tauk,eta,T,k0,kss,h0,hss,g0,gss];
options=optimoptions('fsolve','Display','iter');
sol=fsolve(@(z)foc6(z,param),x0,options);

k= [k0;sol(:,1);kss];
h =[h0;sol(:,2);hss];
g =[g0; sol(:,3);gss];

%From equation (29) we can write
c1 = k(1:end-1).^alpha.*h(1:end-1).^(1-alpha)+(1-delta).*k(1:end-1)-g(1:end-1)-(1+gamma).*(1+eta).*k(2:end);
c = [c1;css];
y = k.^alpha.*h.^(1-alpha);
i = y - g - c;

%Plots, item e
figure 
subplot(2,3,2)
plot(c)
title('Consumption')
subplot(2,3,1)
plot(k)
title('Capital Stock')
subplot(2,3,5)
plot(y)
title('Output')
subplot(2,3,3)
plot(i)
title('Investment')
subplot(2,3,4)
plot(h)
title('Labor')


%% Items f,g

%Initial values and variables in the steady state:
Upsilon = (((1+gamma)-beta*(1-delta))/(beta*alpha*(1-tauk)))^(1/(1-alpha));
Delta = delta +  gamma + eta + eta*gamma;
kss = ((1-alpha)*(1-tauh)*(Upsilon.^(-alpha)))/(theta*(Upsilon.^(1-alpha))*(1-tauh.*(1-alpha) - tauk*alpha) - theta*Delta + (1-alpha).*(1-tauh).*(Upsilon.^(1-alpha)));
hss = Upsilon*kss;
css = (((kss^(alpha))*(hss^(1-alpha))*(1-tauh*(1-alpha) - tauk*alpha)) - kss*Delta)/(1+tauc);
gss = tauc.*css + tauh.*(1-alpha).*(kss.^(alpha)).*(hss.^(1-alpha)) + tauk.*(alpha).*(kss.^(alpha)).*(hss.^(1-alpha));


tkn=0.10; %Tax reform, we are reducing the tax on capital income from 0.15 to 0.10

%Now I calculate the value of the new variables, considering that the tax on labor income is now endogenous, since it will be the one necessary so the can finance the same level of spending

Upsilon = (((1+gamma)-beta*(1-delta))/(beta*alpha*(1-tkn)))^(1/(1-alpha));
Delta = delta +  gamma + eta + eta*gamma;
f=@(hn) hn*Upsilon^(-1) - ((1-alpha)*(1-(1-((1+tauc)*theta*(-gss+Upsilon^(-alpha)*hn-Delta*Upsilon^(-1)*hn))/((1-hn)*Upsilon^(-alpha)*(1-alpha))))*(Upsilon.^(-alpha)))/(theta*(Upsilon.^(1-alpha))*(1-(1-((1+tauc)*theta*(-gss+Upsilon^(-alpha)*hn-Delta*Upsilon^(-1)*hn))/((1-hn)*Upsilon^(-alpha)*(1-alpha))).*(1-alpha) - tkn*alpha) - theta*Delta + (1-alpha).*(1-(1-((1+tauc)*theta*(-gss+Upsilon^(-alpha)*hn-Delta*Upsilon^(-1)*hn))/((1-hn)*Upsilon^(-alpha)*(1-alpha)))).*(Upsilon.^(1-alpha)));
hssn =fsolve(f,0.5);
hn = hssn;
thn=1-((1+tauc)*theta*(-gss+Upsilon^(-alpha)*hn-Delta*Upsilon^(-1)*hn))/((1-hn)*Upsilon^(-alpha)*(1-alpha));
kssn = Upsilon^(-1)*hn;
kn = kssn;
cssn = (((kn^(alpha))*(hn^(1-alpha))*(1-thn*(1-alpha) - tkn*alpha)) - kn*Delta)/(1+tauc);
cn =cssn;
gn = gss;

%Initial conditions
h0=hss;
k0=kss;
c0=css;
g0=gss;

xk=[k0, k0*ones(size(1:T-1))]; 
xh=[h0, h0*ones(size(1:T-1))];
xg=[g0, g0*ones(size(1:T-1))];

x0=[xk',xh',xg'];
param2=[beta,delta,gamma,alpha,theta,tauc,thn,tkn,eta,T,k0,kn,h0,hn,g0,gn];
options=optimoptions('fsolve','Display','iter');
sol2=fsolve(@(z)foc6(z,param2),x0,options);

%Solution
khat=[k0;sol2(:,1);kn]; 
hnn=[h0;sol2(:,2);hn];
ghat=[g0;sol2(:,3);gn];
yhat=khat.^alpha.*hnn.^(1-alpha);
rhatn=(1-tkn).*alpha.*khat.^(alpha-1).*hnn.^(1-alpha);
whatn=(1-alpha).*(khat.^alpha).*(hnn.^(-alpha));
ctn = khat(1:end-1).^alpha.*hnn(1:end-1).^(1-alpha)+(1-delta).*khat(1:end-1)-ghat(1:end-1)-(1+gamma).*(1+eta).*khat(2:end);
chat=[ctn;cn];
ihat = yhat - ghat - chat;

%Plots
subplot(2,3,1)
plot([c;chat])
title('Consumption')
subplot(2,3,2)
plot([k;khat])
title('Capital Stock')
subplot(2,3,3)
plot([y;yhat])
title('Output')
subplot(2,3,4)
plot([i;ihat])
title('Investment')
subplot(2,3,5)
plot([g;ghat])
title('Government spending')
subplot(2,3,6)
plot([h;hnn])
title('Hours worked')

% Tax reform implications
j=@(w)(log((1+w)*cn)+gamma*log(1-hn))-(log(css)+gamma*log(1-hn));
w=fsolve(j,0.5);
%Plot
plot([c;chat])
title('Consumption')
% Compensation must be 1.23%








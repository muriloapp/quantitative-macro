function f=foc6(z,p)

%Vector of Parameters
beta=p(1);
delta=p(2);
gamma=p(3);
alpha=p(4);
theta=p(5);
tauc=p(6);
tauh=p(7);
tauk=p(8);
eta=p(9);   
T=p(10);
k0=p(11);
kss=p(12);
h0=p(13);
hss=p(14);
g0=p(15);
gss=p(16);


%Endogenous variable z
for t=1:T
    k(t)=z(t,1);
    h(t)=z(t,2);
    g(t)=z(t,3);
end

% Steady State
k(T+1)=kss;
h(T+1)=hss;
g(T+1)=gss;

%Initial values 
f1(1)=k(1)^alpha*h(1)^(1-alpha)+(1-delta)*k(1)-g(1)-(1+gamma)*(1+eta)*k(2)-(beta/(1+gamma))*((1-tauk)*alpha*k(1)^(alpha-1)*h(1)^(1-alpha)+1-delta)*(k0^alpha*h0^(1-alpha)+(1-delta)*k0-g0-(1+gamma)*(1+eta)*k(1));
f2(1)=(theta*(k(1)^alpha*h(1)^(1-alpha)+(1-delta)*k(1)-g(1)-(1+gamma)*(1+eta)*k(2)))/(1-h(1))-((1-tauh)/(1+tauc))*(1-alpha)*k(1)^alpha*h(1)^(-alpha);
f3(1)=g(1)-tauc*(k(1)^alpha*h(1)^(1-alpha)+(1-delta)*k(1)-g(1)-(1+gamma)*(1+eta)*k(2))-tauh*(1-alpha)*k(1)^alpha*h(1)^(1-alpha)-tauk*alpha*k(1)^(alpha)*h(1)^(1-alpha);

for t=2:T
    f1(t)=k(t)^alpha*h(t)^(1-alpha)+(1-delta)*k(t)-g(t)-(1+gamma)*(1+eta)*k(t+1)-(beta/(1+gamma))*((1-tauk)*alpha*k(t)^(alpha-1)*h(t)^(1-alpha)+1-delta)*(k(t-1)^alpha*h(t-1)^(1-alpha)+(1-delta)*k(t-1)-g(t-1)-(1+gamma)*(1+eta)*k(t)); 
    f2(t)=(theta*(k(t)^alpha*h(t)^(1-alpha)+(1-delta)*k(t)-g(t)-(1+gamma)*(1+eta)*k(t+1)))/(1-h(t))-((1-tauh)/(1+tauc))*(1-alpha)*k(t)^alpha*h(t)^(-alpha);
    f3(t)=g(t)-tauc*(k(t)^alpha*h(t)^(1-alpha)+(1-delta)*k(t)-g(t)-(1+gamma)*(1+eta)*k(t+1))-tauh*(1-alpha)*k(t)^alpha*h(t)^(1-alpha)-tauk*alpha*k(t)^(alpha)*h(t)^(1-alpha);
end

f=[f1' f2' f3'];






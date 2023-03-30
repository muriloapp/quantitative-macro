function s=model_solution(P,w) 

% Parameters
A0  =  P(1);   
A1  =  P(2);    
alpha    =  P(3);   
sigma  =  P(4);   
beta =  P(5);   
gamma = P(6);
r = P(7);

% Model solution 
w0  = w(1);
w1 = w(2);
L0=(w0/(A0*alpha))^(1/(alpha-1)); %Solution equation (11)
L1=(w1/(A1*alpha))^(1/(alpha-1)); %Solution equation (11)
Y0=L0^alpha; %Solution equation (12)
Y1=L1^alpha; %Solution equation (12)
l1=1-L1; %Solution equation (13)
l0=((w0/w1)*(1+r)*beta)^(-1/sigma)*l1; %Solution equation (8)
c1=(1/(1+(1+r)*((1+r)*beta)^(-1/sigma)))*(w1*(1-l1)+(1+r)*(w0*(1-l0))); %Solving for c1 in the budget constraint
c0=((1+r)*beta)^(-1/sigma)*c1; %Solution equation (15)
a1=w0*(1-l0)-c0; %Solution equation (2), from budget constraint

s=[w0,w1,L0,L1,Y0,Y1,l0,l1,c0,c1,a1];

end
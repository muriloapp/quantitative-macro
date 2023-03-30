function F = foc5(w,P)
% Parameters
A0  =  P(1);   
A1  =  P(2);    
alpha    =  P(3);   
sigma  =  P(4);   
beta =  P(5);   
gamma = P(6);
r = P(7);
% Equations   
%substituting w0 and w1 in the labor market condition
F(1)=(((w(1)/w(2))*(1+r)*beta)^(-1/sigma))*(1-(w(2)/(A1*alpha))^(1/(alpha-1)))-(1-(w(1)/(A0*alpha))^(1/(alpha-1)));
%market clearing and c0+c1=Y0+Y1 condition
F(2)=((1/(1+(1+r)*((1+r)*beta)^(-1/sigma)))*((w(2)*(w(1)/(A0*alpha))^(1/(alpha-1)))+(1+r)*(w(1)*(w(1)/(A0*alpha))^(1/(alpha-1)))))+(1/(1+r))*((1+r)*beta)^(-1/sigma)*((1/(1+(1+r)*((1+r)*beta)^(-1/sigma)))*((w(2)*(w(1)/(A0*alpha))^(1/(alpha-1)))+(1+r)*(w(1)*(w(1)/(A0*alpha))^(1/(alpha-1)))))-(1/(1+r))*((w(2)/(A1*alpha))^(alpha/(alpha-1)))*A1-((w(1)/(A0*alpha))^(alpha/(alpha-1)))*A0;

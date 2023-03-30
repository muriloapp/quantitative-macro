function [f,J] = f2e(z,p)
% c(x) + x = 0
%f = @(x)x^(-5) - x^(-3) + x;

x=z(1);
a=p(1);
b=p(2);
f=x^(a) - x^(b) + x;
J=-5*x^(-6)+3*x^(-4) + 1;
end
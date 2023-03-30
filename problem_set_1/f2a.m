function [f,J] = f2a(z,p)
%f = @(x)x^(-5) - x^(-3) - c;
%f = @(x)x^(a) - x^(b) - c;
x=z(1);
a=p(1);
b=p(2);
c=p(3);
f=x^(a) - x^(b) - c;
J=-5*x^(-6)+3*x^(-4);
end
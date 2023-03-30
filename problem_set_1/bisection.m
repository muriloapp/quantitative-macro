function c = bisection(f,a,b)

 if f(a)*f(b)>0
 disp('Wrong choice for a and b')
 else
 c = (a + b)/2;
 err = abs(f(c));
 while err > 1e-7
 if f(a)*f(c)<0
 b = c;
 else
 a = c;
 end
 c = (a + b)/2;
 err = abs(f(c));
 end
 end

function [yfit, ydfit]=lpk(x,y,h,p,xfit)

% x is input, y is response, h is bandwidth, p is order of lpk
% x and y should be n-by-1 vector

m=length(xfit);
yfit=zeros(m,1);
ydfit=zeros(m,1);

rg=max(x)-min(x);
h=rg*h;

for i=1:m
    T=[];
    for j=1:(p+1)
        T=[T,(x-xfit(i)).^(j-1)/factorial(j-1)];
    end
    ker = max(0.75*(1-(x-xfit(i)).^2/(h^2)),0);
    ker = sqrt(ker/h);
    T_star = T .* repmat(ker, 1, p+1);
    y_star = y .* ker;
    beta=(T_star'*T_star + eye(p+1)*(1e-4))\(T_star'*y_star);
    yfit(i)=beta(1);
    ydfit(i)=beta(2);  
end
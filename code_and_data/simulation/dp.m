% dp.m
function res=dp(theta, lambda, type)
% written by He Jiaxin 
% theta is parameter to be estimated. lambda is the power of penalty. 3
% penalties can be chosen: LASSO, SCAD, MCP. For SCAD, threshold is set as
% 3.7. For MCP, threshold is set as 3.
if strcmpi(type,'LASSO')
res = sign(theta) .* lambda;
elseif strcmpi(type,'SCAD')
a = 3.7;
b1=double(abs(theta)>lambda);
b2=double(abs(theta)<lambda*a);
res=lambda.*(1-b1)+((lambda*a)-abs(theta)).*b2.*b1/(a-1);
elseif strcmpi(type,'MCP')
a = 3;
b = double(abs(theta)<=lambda*a);
res=(lambda - abs(theta)/a).*b;
else
error('Error: The specified penalty type has not been implemented.');
end
end

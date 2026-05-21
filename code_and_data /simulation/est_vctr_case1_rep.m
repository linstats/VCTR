% this is the code to reproduce experiment results in case I.

% in this code, we simulate 2D cases and use kernel smoothing method to 
% estimate function A(t), together with one-way coefficient \beta.
% This code is to repeat our case I.

% set 4 different kind of A(t)


clear all;

% varying coefficient index
n_dict = {2000,5000};    % sample size
n_n = length(n_dict);

% tensor structure
R = 10;
S = 16;

% noise type 
err_dict = {'Gaussian', 'Heavy tailed'};
n_err = length(err_dict);

bR = sqrt((1:R)'/R);
bS = sqrt((1:S)'/S);
At_dist = {@(t) double(full(ktensor({t.^0.5,bR, bS}))), ...  
          @(t) double(full(ktensor({(t-0.5).^2,bR, bS}))), ... 
          @(t) double(full(ktensor({1.75*((exp(-(3*t-1).^2)+exp(-(4*t-3).^2))-0.75),bR, bS}))), ...        
          @(t) double(full(ktensor({sin(2*pi*(t-0.5)),bR, bS})))};         
n_At = length(At_dist);

% True coefficients for regular (non-array) covariates
p0 = 2;
beta = ones(p0,1);

n_rep = 500;

for i_n = 1:n_n
for i_err = 1:n_err
for i_At = 1:n_At
mae_At_ls = zeros(n_rep, 1);
rmse_At_ls = zeros(n_rep, 1);
mae_beta_ls = zeros(n_rep, 1);
rmse_beta_ls = zeros(n_rep, 1);
mae_err_ls = zeros(n_rep, 1);
rmse_err_ls = zeros(n_rep, 1);
for i_rep = 1:n_rep
if mod(i_rep,10)==0
fprintf('This is %d-th repeat.\n', i_rep)
end

n = n_dict{i_n};    % sample size
t = sort(rand(n,1));    % index variable, t \in [0,1]

X = randn(n,R,S);  % n-by-R-by-S matrix variates
Z = randn(n,p0);   % n-by-p0 regular design matrix
At = At_dist{i_At}(t);
mu = sum(X .* At, [2,3]) + Z * beta;

err_type = err_dict{i_err};
switch err_type
    case 'Gaussian' 
        err = randn(n,1);
    case 'Heavy tailed'
        err = trnd(5,n,1);   
    otherwise
        error('Invalid noise selection.');
end
y = mu + err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose several time point t to estimate \hat{a}(t)
n_est = 0.2*n;
id_est = sort(n*0.15 + randperm(n*0.7, n_est))';
t_est = t(id_est);

y_est = y(id_est);
X_est = X(id_est, :, :);
Z_est = Z(id_est, :);

At_est = At(id_est,:,:);
beta_est = beta;

% prepare Z and X for regression
At_hat = zeros(n_est, R, S);
h1 = 0.13;
for i=1:n_est
ker = max(.75*(1-(t-t_est(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

y_star = y .* ker;
Z_star = Z .* repmat(ker, 1, p0);
X_mat = reshape(X, n, R*S);
X_mat_a = X_mat;
sst = (t - t_est(i))/h1;
X_mat_b = X_mat_a .* repmat(sst, 1, R*S);

X_mat_a = X_mat_a .* repmat(ker, 1, R*S);
X_mat_b = X_mat_b .* repmat(ker, 1, R*S);

XZ_star = [X_mat_a, X_mat_b, Z_star];
para_hat = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_hat_i = para_hat(1:R*S);
beta_hat_i = para_hat(2*R*S+1:end);
At_hat(i,:,:) = reshape(At_hat_i, R,S);
end

yte = y_est - sum(X_est .* At_hat, [2,3]);
beta_hat = (Z_est' * Z_est + eye(p0) * (1e-4))\(Z_est' * yte);

y_hat = sum(X_est .* At_hat, [2,3]) + Z_est * beta_hat;
err_hat = y_est - y_hat;

mae_At = mean(abs(At_est - At_hat),'all');
rmse_At = sqrt(mean((At_est - At_hat).^2,'all'));
mae_beta = mean(abs(beta_hat-beta_est));
rmse_beta = sqrt(mean((beta_hat-beta_est).^2));
mae_err = mean(abs(err_hat),'all');
rmse_err = sqrt(mean(err_hat.^2, 'all'));

mae_At_ls(i_rep) = mae_At;
rmse_At_ls(i_rep) = rmse_At;
mae_beta_ls(i_rep) = mae_beta;
rmse_beta_ls(i_rep) = rmse_beta;
mae_err_ls(i_rep) = mae_err;
rmse_err_ls(i_rep) = rmse_err;
end
fprintf(['This is repeated experiment for case 1,\n' ...
    'sample size n = %d, error type is %s, %d-th At function.\n' ...
    'MAE for At: %.4f(%.4f), RMSE for At: %.4f(%.4f),\n' ...
    'MAE for beta: %.4f(%.4f), RMSE for beta: %.4f(%.4f),\n' ...
    'MAE for error: %.4f(%.4f), RMSE for error: %.4f(%.4f).\n'], ...
    n, err_dict{i_err}, i_At, ...
    mean(mae_At_ls), std(mae_At_ls), mean(rmse_At_ls), std(rmse_At_ls),...
    mean(mae_beta_ls), std(mae_beta_ls), mean(rmse_beta_ls), std(rmse_beta_ls),...
    mean(mae_err_ls), std(mae_err_ls), mean(rmse_err_ls), std(rmse_err_ls));
end
end
end
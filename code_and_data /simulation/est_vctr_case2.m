% This is the code of estimation parameters of simulation study - case 2.

clear all;

% varying coefficient index
n = 2000;    % sample size
t = sort(rand(n,1)); % index variable

% 2D varying coefficient
R=5;
S=64;
bR = sqrt((1:R)'/R);
bS = sqrt((1:S)'/S);
At = full(ktensor({4*(t-0.5).^2,bR, bS}));
% At = full(ktensor({t.^0.5,bR, bS}));
% At = full(ktensor({1.75*((exp(-(3*t-1).^2)+exp(-(4*t-3).^2))-0.75),bR, bS}));
% At = full(ktensor({sin(6*pi*(t-0.5)),bR, bS}));
At = double(At);

X = randn(n,R,S);  % n-by-R-by-S matrix variates

% True coefficients for regular (non-array) covariates
p0 = 2;
beta = 3*ones(p0,1);

Z = randn(n,p0);   % n-by-p0 regular design matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = sum(X .* At, [2,3]) + Z * beta;
err= randn(n,1);

y = mu + err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Least square estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose several time point t to estimate \hat{a}(t)
n_est = 0.1*n;
id_est = sort(n*0.2 + randperm(n*0.6, n_est))';
t_est = t(id_est);

y_est = y(id_est);
X_est = X(id_est, :, :);
Z_est = Z(id_est, :);

At_est = At(id_est,:,:);
beta_est = beta;

% prepare Z and X for regression
At_hat = zeros(n_est, R, S);
h1 = 0.2;
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
mae = mean(abs(err_hat),'all');
rmse = sqrt(mean(err_hat.^2, 'all'));

fprintf('MAE for At: %.4f, RMSE for At: %.4f\n', mae_At, rmse_At);
fprintf('MAE for beta: %.4f, RMSE for beta: %.4f\n', mae_beta, rmse_beta);
fprintf('MAE for error: %.4f, RMSE for error: %.4f\n', mae, rmse);



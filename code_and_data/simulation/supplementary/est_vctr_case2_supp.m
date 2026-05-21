% code to reproduce experiment result in supplementary Case II 


clear all;

% varying coefficient index
n_dict = {1200,1600};    % sample size
n_n = length(n_dict);

% 2D varying coefficient
% strt_dict = {[6,27], [3,27], [3,64]};
strt_dict = {[6,64]};
n_strt = length(strt_dict);

for i_strt = 1:n_strt
strt = strt_dict{i_strt};
R = strt(1); S = strt(2);

bR = sqrt((1:R)'/R);
bS = sqrt((1:S)'/S);
At_dist = {@(t) double(full(ktensor({t.^0.5,bR, bS}))), ...  
          @(t) double(full(ktensor({(t-0.5).^2,bR, bS}))), ... 
          @(t) double(full(ktensor({1.75*((exp(-(3*t-1).^2)+exp(-(4*t-3).^2))-0.75),bR, bS}))), ...        
          @(t) double(full(ktensor({sin(2*pi*(t-0.5)),bR, bS})))};         
n_At = length(At_dist);

% one-way constant coefficient
p0 = 2;
beta = ones(p0,1);

% noise type 
err_dict = {'Gaussian', 'Heavy tailed'};
n_err = length(err_dict);

% rules of thumb
% h1 = 1.06 / sqrt(12) * n^(-1/5);
h1 = 0.2;

for i_n = 1:n_n
for i_err = 1:n_err
for i_At = 1:n_At

% repeat times
n_rep = 10;
rmise_At_rep = zeros(n_rep,1);
rmse_beta_rep = zeros(n_rep,1);
rmse_err_rep = zeros(n_rep,1);
for i_rep = 1:n_rep

n = n_dict{i_n};        % sample size
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
id_est = sort(n*0.2 + randperm(n*0.6, n_est))';
t_est = t(id_est);

y_est = y(id_est);
X_est = X(id_est, :, :);
Z_est = Z(id_est, :);

At_est = At(id_est,:,:);
beta_est = beta;

At_hat = zeros(n_est, R, S);
for i=1:n_est

Kh = @(u,h) (1./(sqrt(2*pi)*h)) .* exp(-0.5*(u./h).^2);
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
para_hat = (XZ_star' * XZ_star + 1e-4 * eye(2*R*S+p0)) \ XZ_star' * y_star;
At_hat_i = para_hat(1:R*S);
At_hat(i,:,:) = reshape(At_hat_i, R,S);
end

yte = y_est - sum(X_est .* At_hat, [2,3]);
beta_hat = (Z_est' * Z_est + eye(p0) * (1e-4))\(Z_est' * yte);

y_hat = sum(X_est .* At_hat, [2,3]) + Z_est * beta_hat;
err_hat = y_est - y_hat;

rmise_At = sqrt(mean((At_est - At_hat).^2,'all'));
rmse_beta = sqrt(mean((beta_hat - beta_est).^2));
rmse_err = sqrt(mean(err_hat.^2, 'all'));

rmise_At_rep(i_rep) = rmise_At;
rmse_beta_rep(i_rep) = rmse_beta;
rmse_err_rep(i_rep) = rmse_err;
end

fprintf('CP rank = %d, number of blocks = %d, Sample size n = %d, error type is %s, %d-th At function.\n', ...
        R, S, n_dict{i_n}, err_dict{i_err}, i_At);
fprintf('RMISE for At: %.4f(%.4f), RMSE for beta: %.4f(%.4f), RMSE for error: %.4f(%.4f).\n', ...
    mean(rmise_At_rep), std(rmise_At_rep), mean(rmse_beta_rep), std(rmse_beta_rep), mean(rmse_err_rep), std(rmse_err_rep));
end
end
end
end


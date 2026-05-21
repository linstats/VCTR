% This is the code of evaluate parameters of simulation study - case 4.
% except the process in est_vctr_case3, we further use the result of model
% identification to do a refined regression. 
% there are total 3 steps: 1. orcacle kernel smoothing regression, 2.
% penalized spline regression, 3. refined kernel smoothing regression

% we set different n = 2000 and 5000 

clear all;

% set hyper parameters
h2 = 0.15;

dif_thrs = 1e-1;
ind_thrs = 1e-5;
tot_iter = 100;

% pentype = 'LASSO';
pentype = 'SCAD'; 
% pentype = 'MCP';

if strcmp(pentype, 'LASSO') 
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.005;
elseif strcmp(pentype, 'SCAD')
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.07;
elseif strcmp(pentype, 'MCP')
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.07;
else
error('no such penalty');
end

% varying coefficient index
n = 2000;    % sample size
t = rand(n,1); % index variable
[t, ~] = sort(t);

% 2D varying coefficient
R=20;
S=64;
sp = 10;
bR = (1:R)'/R;
bS = (1:S)'/S;

At=zeros(n,R,S);
for r = 1:sp
for s = 1:sp
if s < r 
    At(:,r,s) = sin(2*pi*(t-0.5)) * sqrt(bR(r)) * sqrt(bS(s));
else 
    At(:,r,s) = sqrt(bR(r)) * sqrt(bS(s));
end
end
end

%{
sur_t = [1,100:100:5000];
isosurface(At(sur_t, :,:))
%}

%%%%%%%%%%%%%%%%%%%%%%%%
% True coefficients for regular one-way (non-array) covariates
p0 = 5;
beta = [1,1,0,0,0]';

%%%%%%%%%%%%%%%%%%%%%%%%
% construct X tensor variates and Z one-way covariates
% correlation between tensor covariates, first order auto-regressive covariance
X = randn(n,R,S);  % R-by-S-by-n matrix variates
Z = randn(n,p0);   % n-by-p0 regular design matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = sum(X .* At, [2,3]) + Z * beta;

err= randn(n,1);
y = mu + err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B-spline projection
% here we use all time points in spline model
q = 4; n_knot = 6;
knots = linspace(0,1,n_knot);
Bst=bspline_basismatrix(q,[zeros(1,q-1), knots, ones(1,q-1)], t);
L=size(Bst,2);

%spline
Bs = reshape(Bst,[n,1,1,L]);
X_Bs = X .* Bs;
X_Bs = permute(X_Bs, [4,2,3,1]); % L-R-S-n
%tic;
[beta_hat_init,gamma_hat_cp_init,glmstats1] = kruskal_reg(Z,tensor(X_Bs),y,2,'normal');
% toc;
gamma_hat_init=double(full(gamma_hat_cp_init)); % L-R-S

At_hat_init = double(ttt(tensor(Bst), tensor(gamma_hat_init),2,1));  % n-R-S

y_hat_init = sum(X .* At_hat_init, [2,3]) + Z * beta_hat_init;
err_hat_init = y - y_hat_init;

mean(abs(err_hat_init),'all')
sqrt(mean(err_hat_init.^2, 'all'))

%{
mae_At = mean(abs(At - At_hat_init),'all');
rmse_At = sqrt(mean((At - At_hat_init).^2,'all'));
mae_beta = mean(abs(beta_hat_init-beta));
rmse_beta = sqrt(mean((beta_hat_init-beta).^2));
mae = mean(abs(err_hat_init),'all');
rmse = sqrt(mean(err_hat_init.^2, 'all'));

fprintf('MAE for At: %.4f, RMSE for At: %.4f\n', mae_At, rmse_At);
fprintf('MAE for beta: %.4f, RMSE for beta: %.4f\n', mae_beta, rmse_beta);
fprintf('MAE for error: %.4f, RMSE for error: %.4f\n', mae, rmse);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add penalty to distinguish const_0, const_not0, and varying coefficient
% in gamma and beta
tic;

% step 1: initialize gamma, beta and penalty weight 
gamma_old = gamma_hat_init;
beta_old = beta_hat_init;

dif=1;
iter=0;
while dif>dif_thrs && iter<tot_iter 
iter=iter+1;
% step2: given gamma, update beta
y_tilde = y - double(ttt(tensor(X_Bs), tensor(gamma_old), [1,2,3], [1,2,3])); % n-by-1
omega_beta = diag(dp(beta_old,penparam_beta,pentype)./(abs(beta_old)+1e-10));
left=Z' * Z + n/2 * omega_beta;
right=Z' * y_tilde;
beta_new = left \ right;

% step3: given beta, update gamma
% update by part, in each part, update gamma_rs=[gamma_1rs, gamma_2rs, ..,
% gamma_Lrs] L parameters. So here are total RS parts
gamma_new=gamma_old;
for s=1:S
gamma_s_old=squeeze(gamma_old(:,:,s));
y_tilde = y - double(ttt(tensor(X_Bs), tensor(gamma_new), [1,2,3], [1,2,3])) - Z * beta_new...
    + double(ttt(tensor(squeeze(X_Bs(:,:,s,:))), tensor(gamma_s_old), [1,2],[1,2])); % n-by-1
X_Bs_tilde = double(reshape(X_Bs(:,:,s,:), [L*R, n]))'; %n-by-LR
omega_gamma_c = zeros(L*R, L*R);
omega_gamma_v = zeros(L*R, L*R);
for r = 1:R
gamma_rs_old = squeeze(gamma_s_old(:,r));
omega_gamma_c((r-1)*L+1:r*L, (r-1)*L+1:r*L) = dp(abs(mean(gamma_rs_old)),penparam_gamma_c, pentype)...
    /(abs(mean(gamma_rs_old))+1e-10) * 1/(L*L)*ones(L,L);
omega_gamma_v((r-1)*L+1:r*L, (r-1)*L+1:r*L) = dp(norm(gamma_rs_old-mean(gamma_rs_old)),penparam_gamma_v, pentype)...
    / (norm(gamma_rs_old-mean(gamma_rs_old))+1e-10) * (eye(L)-1/L*ones(L,L));
end
left = (X_Bs_tilde')*X_Bs_tilde + n/2*omega_gamma_c + n/2*omega_gamma_v;
right = (X_Bs_tilde')*y_tilde;
gamma_s_new=left\right;
%[a_rs_new, a_rs_old]
gamma_new(:,:,s)=reshape(gamma_s_new,L,R);
end

dif=sqrt(mean((gamma_new - gamma_old).^2, 'all'));
gamma_old=gamma_new;
beta_old = beta_new;
end

gamma_hat_pen = gamma_old;
beta_hat_pen = beta_old;
At_hat_pen = double(ttt(tensor(Bst), tensor(gamma_hat_pen), 2,1));

toc;

y_hat_pen = sum(X .* At_hat_pen, [2,3]) + Z * beta_hat_pen;
err_hat_pen = y - y_hat_pen;

mean(abs(err_hat_pen),'all')
sqrt(mean(err_hat_pen.^2, 'all'))

%{
mae_At = mean(abs(At - At_hat_pen),'all');
rmse_At = sqrt(mean((At - At_hat_pen).^2,'all'));
mae_beta = mean(abs(beta_hat_pen-beta));
rmse_beta = sqrt(mean((beta_hat_pen-beta).^2));
mae = mean(abs(err_hat_pen),'all');
rmse = sqrt(mean(err_hat_pen.^2, 'all'));

fprintf('MAE for At: %.4f, RMSE for At: %.4f\n', mae_At, rmse_At);
fprintf('MAE for beta: %.4f, RMSE for beta: %.4f\n', mae_beta, rmse_beta);
fprintf('MAE for error: %.4f, RMSE for error: %.4f\n', mae, rmse);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find out zero constant coefficient, non-zeros constant coefficient and
% varying coefficients.
abs_beta_pen = abs(beta_hat_pen);
At_pen_c = squeeze(abs(mean(At_hat_pen,1)));
At_pen_v = squeeze(sqrt(mean((At_hat_pen - mean(At_hat_pen,1)).^2,1)));

flag_beta_nonzero = abs_beta_pen > ind_thrs;

flag_At_const_zero = (At_pen_v < ind_thrs) & (At_pen_c < ind_thrs);
flag_At_const_nonzero = (At_pen_v < ind_thrs) & (At_pen_c >= ind_thrs);
flag_At_vary = At_pen_v >= ind_thrs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Least sqaure estimation
% construct y_vec, X_vary, X_const_nonzero, Z_nonzero do kernelized varying-coefficient
% regression

X_mat = reshape(X, n, R*S);  % n-by-R*S

%constant coefficient
beta_nonzero_id = find(flag_beta_nonzero==1);
Z_nonzero = Z(:,beta_nonzero_id);

[const_row_idx, const_col_id] = find(flag_At_const_nonzero==1);
const_id = sub2ind(size(At_hat_pen,2:3), const_row_idx, const_col_id);
X_const_nonzero = X_mat(:,const_id);

XZ_const_nonzero = [X_const_nonzero, Z_nonzero];

%varying coefficient
[vary_row_id, vary_col_id] = find(flag_At_vary==1);
vary_id = sub2ind(size(At_hat_pen,2:3), vary_row_id, vary_col_id);
X_vary = X_mat(:,vary_id);

% select several time point to estimate A(t) and beta 
n_est = 0.1*n;
id_est = sort(n*0.15 + randperm(n*0.7, n_est))';
t_est = t(id_est);

y_est = y(id_est);
X_est = X(id_est, :, :);
Z_est = Z(id_est, :);

Z_nonzero_est = Z_nonzero(id_est, :);
X_const_nonzero_est = X_const_nonzero(id_est, :);
XZ_const_nonzero_est = [X_const_nonzero_est, Z_nonzero_est];

X_vary_est = X_vary(id_est,:);

At_est = At(id_est,:,:);
beta_est = beta;

At_vary_hat = zeros(n_est, length(vary_id));
for i=1:n_est
ker = max(.75*(1-(t-t_est(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

y_star = y .* ker;
XZ_const_nonzero_star = XZ_const_nonzero .* repmat(ker, 1, length(beta_nonzero_id) + length(const_id));

X_vary_a = X_vary;
sst = (t - t_est(i))/h2;
X_vary_b = X_vary_a .* repmat(sst, 1, length(vary_id));

X_vary_a = X_vary_a .* repmat(ker, 1, length(vary_id));
X_vary_b = X_vary_b .* repmat(ker, 1, length(vary_id));

XZ_star = [X_vary_a, X_vary_b, XZ_const_nonzero_star];
para_hat_ref = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_vary_hat_i = para_hat_ref(1:length(vary_id));
At_vary_hat(i,:) = At_vary_hat_i;
end

yte = y_est - sum(X_vary_est .* At_vary_hat,2);
para_hat_ref = (XZ_const_nonzero_est' * XZ_const_nonzero_est + eye(length(beta_nonzero_id) + length(const_id)) * (1e-4))...
                \ (XZ_const_nonzero_est' * yte);

At_const_nonzero_hat = para_hat_ref(1:length(const_id));
At_const_nonzero_hat = repmat(At_const_nonzero_hat',n_est,1);

At_hat_ref = zeros(n_est,R*S);
At_hat_ref(:, vary_id) = At_vary_hat;
At_hat_ref(:, const_id) = At_const_nonzero_hat;
At_hat_ref = reshape(At_hat_ref, n_est,R,S);

beta_nonzero_hat = para_hat_ref(end-length(beta_nonzero_id)+1:end);
beta_hat_ref = zeros(p0,1);
beta_hat_ref(beta_nonzero_id) = beta_nonzero_hat;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute estimated error and do refined regression
y_hat_ref = sum(X_est .* At_hat_ref, [2,3]) + Z_est * beta_hat_ref;
err_hat_ref = y_est - y_hat_ref;

mean(abs(err_hat_ref),'all')
sqrt(mean(err_hat_ref.^2, 'all'))

%{
mae_At = mean(abs(At_est - At_hat_ref),'all');
rmse_At = sqrt(mean((At_est - At_hat_ref).^2,'all'));
mae_beta = mean(abs(beta_est - beta_hat_ref));
rmse_beta = sqrt(mean((beta_est - beta_hat_ref).^2));
mae = mean(abs(err_hat_ref),'all');
rmse = sqrt(mean(err_hat_ref.^2, 'all'));

fprintf('MAE for At: %.4f, RMSE for At: %.4f\n', mae_At, rmse_At);
fprintf('MAE for beta: %.4f, RMSE for beta: %.4f\n', mae_beta, rmse_beta);
fprintf('MAE for error: %.4f, RMSE for error: %.4f\n', mae, rmse);
%}



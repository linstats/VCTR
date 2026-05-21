% This is the code of evaluate se, ppv, spe, npv of simulation study - case 3.
% in case 3, we consider an A(t) which contains constant nonzero and zero
% functions, and an beta which contains sparse parameters. Here we use 
% penalty regression to identify those functions. We make sure this is not
% an overfitting problem, and an original kernel smoothing regression can
% be implemented.


clear all;

% covariance parameters
rho=0.1; 

% set hyper parameters
% pentype = 'LASSO';
pentype = 'SCAD';
% pentype = 'MCP';

ind_thrs = 1e-1;
dif_thrs = 1e-4;
tot_iter = 100;

if strcmp(pentype, 'LASSO') && rho == 0.1
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.01;
elseif strcmp(pentype, 'LASSO') && rho ~= 0.1
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.01;
elseif strcmp(pentype, 'SCAD')
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.07;
elseif strcmp(pentype, 'MCP') && rho == 0.1
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.08;
elseif strcmp(pentype, 'MCP') && rho ~= 0.1
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.09;
else
error('no such penalty');
end

% varying coefficient index
n = 5000;    % sample size
t = rand(n,1); % index variable
[t, ~] = sort(t);

% 2D varying coefficient
R=20;
S=16;
sp = 10;
bR = sqrt((1:R)'/R);
bS = sqrt((1:S)'/S);

At=zeros(n,R,S);
for r = 1:sp
for s = 1:sp
if s < r 
    At(:,r,s) = sin(2*pi*(t-0.5)) * bR(r) * bS(s);
else 
    At(:,r,s) = bR(r) * bS(s);
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
sii=1;
% rho=0.1; 
arxx=zeros(sqrt(S), sqrt(S));
for i=1:sqrt(S)
    for j=1:sqrt(S)
        arxx(i,j)=sii*rho^(abs(i-j));
    end
end
crx2=sqrtm(arxx);
crx4=sqrtm(crx2);

X = randn(n,R,sqrt(S),sqrt(S));  % R-by-S-by-n matrix variates
for i = 1:n
for r = 1:R
X(i,r,:,:)=tensor(crx4*squeeze(X(i,r,:,:))*crx4);
end
end
X = reshape(X,[n,R,S]);

% Simulate covariates
Z = randn(n,p0);   % n-by-p0 regular design matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = sum(X .* At, [2,3]) + Z * beta;

err= randn(n,1);
y = mu + err;
%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Least square estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare Z and X for regression
n_est = 0.1*n;
id_est = sort(n*0.15 + randperm(n*0.7, n_est))';
t_est = t(id_est);

y_est = y(id_est);
X_est = X(id_est, :, :);
Z_est = Z(id_est, :);

At_est = At(id_est,:,:);
beta_est = beta;

% prepare Z and X for regression
At_hat = zeros(n_est, R, S);
h1 = 0.15;
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

mean(abs(err_hat),'all')
sqrt(mean(err_hat.^2, 'all'))

%{
mae_At = mean(abs(At - At_hat),'all');
rmse_At = sqrt(mean((At - At_hat).^2,'all'));
mae_beta = mean(abs(beta_hat-beta));
rmse_beta = sqrt(mean((beta_hat-beta).^2));
mae = mean(abs(err_hat),'all');
rmse = sqrt(mean(err_hat.^2, 'all'));

fprintf('MAE for At: %.4f, RMSE for At: %.4f\n', mae_At, rmse_At);
fprintf('MAE for beta: %.4f, RMSE for beta: %.4f\n', mae_beta, rmse_beta);
fprintf('MAE for error: %.4f, RMSE for error: %.4f\n', mae, rmse);
%}
%}
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

flag_At_vary = zeros(size(At_hat_pen,[2,3]));
flag_At_const_nonzero = zeros(size(At_hat_pen,[2,3]));
for r=1:R
for s=1:S
if At_pen_v(r,s) >= ind_thrs
    flag_At_vary(r,s) = 1;
elseif At_pen_c(r,s) >= ind_thrs
    flag_At_const_nonzero(r,s) = 1;
end
end
end
flag_At_const_zero = ones(size(At_hat_pen,[2,3])) - flag_At_const_nonzero - flag_At_vary;

%{
flag_At_const_zero(1:20, 1:20)
flag_At_const_nonzero(1:20, 1:20)
flag_At_vary(1:20, 1:20)
flag_beta_nonzero'
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after sparse regression, identify zero const cells, constant cells
abs_beta_pen = abs(beta_hat_pen);
At_pen_c = squeeze(abs(mean(At_hat_pen,1)));
At_pen_v = squeeze(sqrt(mean((At_hat_pen - mean(At_hat_pen,1)).^2,1)));

abs_beta = abs(beta);
At_c = squeeze(abs(mean(At,1)));
At_v = squeeze(sqrt(mean((At - mean(At,1)).^2,1)));

%{
At_pen_c(1:10, 1:10)
At_pen_v(1:10, 1:10)
At_c(1:10, 1:10)
At_v(1:10, 1:10)
%}

% nonzero beta 
beta_nonzero_ind = abs_beta_pen > ind_thrs;
true_beta_nonzero_ind = abs_beta > ind_thrs;
se_beta_nonzero=sum(beta_nonzero_ind.*true_beta_nonzero_ind, 'all')/sum(true_beta_nonzero_ind, 'all');
ppv_beta_nonzero=sum(beta_nonzero_ind.*true_beta_nonzero_ind, 'all')/sum(beta_nonzero_ind, 'all');
spe_beta_nonzero=sum((1-beta_nonzero_ind).*(1-true_beta_nonzero_ind), 'all')/sum(1-true_beta_nonzero_ind, 'all');
npv_beta_nonzero=sum((1-beta_nonzero_ind).*(1-true_beta_nonzero_ind),'all')/sum(1-beta_nonzero_ind, 'all');

% constant zero coefficient
const_zero_ind = (At_pen_v < ind_thrs) & (At_pen_c < ind_thrs);
true_const_zero_ind = (At_v < ind_thrs) & (At_c < ind_thrs);
se_const_zero=sum(const_zero_ind.*true_const_zero_ind, 'all')/sum(true_const_zero_ind, 'all');
ppv_const_zero=sum(const_zero_ind.*true_const_zero_ind, 'all')/sum(const_zero_ind, 'all');
spe_const_zero=sum((1-const_zero_ind).*(1-true_const_zero_ind), 'all')/sum(1-true_const_zero_ind, 'all');
npv_const_zero=sum((1-const_zero_ind).*(1-true_const_zero_ind),'all')/sum(1-const_zero_ind, 'all');

% const non-zero coefficient 
const_nonzero_ind = (At_pen_v < ind_thrs) & (At_pen_c >= ind_thrs);
true_const_nonzero_ind = (At_v < ind_thrs) & (At_c >= ind_thrs);
se_const_nonzero=sum(const_nonzero_ind.*true_const_nonzero_ind, 'all')/sum(true_const_nonzero_ind, 'all');
ppv_const_nonzero=sum(const_nonzero_ind.*true_const_nonzero_ind, 'all')/sum(const_nonzero_ind, 'all');
spe_const_nonzero=sum((1-const_nonzero_ind).*(1-true_const_nonzero_ind), 'all')/sum(1-true_const_nonzero_ind, 'all');
npv_const_nonzero=sum((1-const_nonzero_ind).*(1-true_const_nonzero_ind), 'all')/sum(1-const_nonzero_ind, 'all');

% varying coefficient
vary_ind= At_pen_v >= ind_thrs;
true_vary_ind = At_v >= ind_thrs;
se_vary=sum(vary_ind.*true_vary_ind, 'all')/sum(true_vary_ind, 'all');
ppv_vary=sum(vary_ind.*true_vary_ind, 'all')/sum(vary_ind, 'all');
spe_vary=sum((1-vary_ind).*(1-true_vary_ind), 'all')/sum(1-true_vary_ind, 'all');
npv_vary=sum((1-vary_ind).*(1-true_vary_ind), 'all')/sum(1-vary_ind, 'all');

%{
const_zero_ind(1:10, 1:10)
const_nonzero_ind(1:10, 1:10)
vary_ind(1:10, 1:10)

true_const_zero_ind(1:10, 1:10)
true_const_nonzero_ind(1:10, 1:10)
true_vary_ind(1:10, 1:10)
%}

fprintf('value of se, ppv, spe, npv with constant zero coefficients are %.2f,%.2f,%.2f,%.2f,\n', se_const_zero, ppv_const_zero, spe_const_zero, npv_const_zero);
fprintf('value of se, ppv, spe, npv with constant non-zero coefficients are %.2f,%.2f,%.2f,%.2f,\n', se_const_nonzero, ppv_const_nonzero, spe_const_nonzero, npv_const_nonzero);
fprintf('value of se, ppv, spe, npv with varying coefficients are %.2f,%.2f,%.2f,%.2f,\n', se_vary, ppv_vary, spe_vary, npv_vary);

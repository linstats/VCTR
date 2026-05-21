

clear all;

% set hyper parameters
h2 = 0.13;
dif_thrs = 1e-5;
ind_thrs = 1e-1;
tot_iter = 100;

% 2D varying coefficient
R=20;
S=64;
sp = 10;
bR = sqrt((1:R)'/R);
bS = sqrt((1:S)'/S);

ns = [2000,5000];
err_hat_gt = cell(2,1);
err_hat_gt{1,1} = zeros(ns(1),1); err_hat_gt{2,1} = zeros(ns(2),1);
err_hat_lasso = cell(2,1);
err_hat_lasso{1,1} = zeros(ns(1),1); err_hat_lasso{2,1} = zeros(ns(2),1);
err_hat_scad = cell(2,1);
err_hat_scad{1,1} = zeros(ns(1),1); err_hat_scad{2,1} = zeros(ns(2),1);
err_hat_mcp = cell(2,1);
err_hat_mcp{1,1} = zeros(ns(1),1); err_hat_mcp{2,1} = zeros(ns(2),1);

for i_n = 1:length(ns)
n = ns(i_n);

% varying coefficient index
t = rand(n,1); % index variable
[t, ~] = sort(t);

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
X = randn(n,R,S);  % R-by-S-by-n matrix variates
Z = randn(n,p0);   % n-by-p0 regular design matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = sum(X .* At, [2,3]) + Z * beta;

err= randn(n,1);
y = mu + err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% divide the whole dataset into train and test sets
k_fold = 10;
CVP=cvpartition(n,'kFold',k_fold); %% 10-fold cv

err_hat = zeros(n,1);
for i_fold = 1:k_fold
fprintf('This is %d-th fold for sample size %d.\n', i_fold, n);

n_train = CVP.TrainSize(i_fold);
n_test = CVP.TestSize(i_fold);

train_flag = training(CVP,i_fold);
test_flag=test(CVP,i_fold);

t_train = t(train_flag);
t_test = t(test_flag);

y_train=y(train_flag,:);
y_test=y(test_flag,:);

X_train = X(train_flag,:,:);
X_test = X(test_flag,:,:);

Z_train = Z(train_flag,:);
Z_test = Z(test_flag,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% oracle VCTR model:
% In this situation, we know the true model structure. So we first find out 
% zero constant coefficient, non-zeros constant coefficient and varying 
% coefficients.

abs_beta = abs(beta);
At_c = squeeze(abs(mean(At,1)));
At_v = squeeze(sqrt(mean((At - mean(At,1)).^2,1)));

flag_beta_nonzero = abs_beta > ind_thrs;

flag_At_const_zero = (At_v < ind_thrs) & (At_c < ind_thrs);
flag_At_const_nonzero = (At_v < ind_thrs) & (At_c >= ind_thrs);
flag_At_vary = At_v >= ind_thrs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Least sqaure estimation
% construct y_vec, X_vary, X_const_nonzero, Z_nonzero do kernelized varying-coefficient
% regression

X_mat_train = reshape(X_train, n_train, R*S);  % n-by-R*S
X_mat_test = reshape(X_test, n_test, R*S);  % n-by-R*S

%constant coefficient
beta_nonzero_id = find(flag_beta_nonzero==1);
Z_nonzero_train = Z_train(:,beta_nonzero_id);
Z_nonzero_test = Z_test(:,beta_nonzero_id);

[const_row_idx, const_col_id] = find(flag_At_const_nonzero==1);
const_id = sub2ind(size(At,2:3), const_row_idx, const_col_id);
X_const_nonzero_train = X_mat_train(:,const_id);
X_const_nonzero_test = X_mat_test(:,const_id);

XZ_const_nonzero_train = [X_const_nonzero_train, Z_nonzero_train];
XZ_const_nonzero_test = [X_const_nonzero_test, Z_nonzero_test];

%varying coefficient
[vary_row_id, vary_col_id] = find(flag_At_vary==1);
vary_id = sub2ind(size(At,2:3), vary_row_id, vary_col_id);
X_vary_train = X_mat_train(:,vary_id);
X_vary_test = X_mat_test(:,vary_id);

At_vary_hat_test = zeros(n_test, length(vary_id));
use_flag_ref_test = true(n_test,1);
for i=1:n_test
ker = max(.75*(1-(t_train-t_test(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

if sum(ker > 0) <= 1.1*(2*length(vary_id)+length(const_id)+length(beta_nonzero_id))
    use_flag_ref_test(i) = false;
    continue
end

y_star = y_train .* ker;
XZ_const_nonzero_star = XZ_const_nonzero_train .* repmat(ker, 1, length(const_id)+length(beta_nonzero_id));

X_vary_a = X_vary_train;
sst = (t_train - t_test(i))/h2;
X_vary_b = X_vary_a .* repmat(sst, 1, length(vary_id));

X_vary_a = X_vary_a .* repmat(ker, 1, length(vary_id));
X_vary_b = X_vary_b .* repmat(ker, 1, length(vary_id));

XZ_star = [X_vary_a, X_vary_b, XZ_const_nonzero_star];
para_hat_ref = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_vary_hat_i = para_hat_ref(1:length(vary_id));
At_vary_hat_test(i,:) = At_vary_hat_i;
end

% estimation beta for Z without information of test set.
n_est = 0.1*n_train;
id_est = randperm(n_train, n_est)';
t_est = t_train(id_est);

y_est = y_train(id_est);
Z_nonzero_est = Z_nonzero_train(id_est, :);
X_const_nonzero_est = X_const_nonzero_train(id_est, :);
XZ_const_nonzero_est = [X_const_nonzero_est, Z_nonzero_est];
X_vary_est = X_vary_train(id_est,:);

At_vary_hat_est = zeros(n_est, length(vary_id));
use_flag_ref_est = true(n_est,1);
for i=1:n_est
ker = max(.75*(1-(t_train-t_est(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

if sum(ker > 0) <= 1.1*(2*length(vary_id)+length(const_id)+length(beta_nonzero_id))
    use_flag_ref_est(i) = false;
    continue
end

y_star = y_train .* ker;
XZ_const_nonzero_star = XZ_const_nonzero_train .* repmat(ker, 1, length(const_id)+length(beta_nonzero_id));

X_vary_a = X_vary_train;
sst = (t_train - t_est(i))/h2;
X_vary_b = X_vary_a .* repmat(sst, 1, length(vary_id));

X_vary_a = X_vary_a .* repmat(ker, 1, length(vary_id));
X_vary_b = X_vary_b .* repmat(ker, 1, length(vary_id));

XZ_star = [X_vary_a, X_vary_b, XZ_const_nonzero_star];
para_hat_ref = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_vary_hat_i = para_hat_ref(1:length(vary_id));
At_vary_hat_est(i,:) = At_vary_hat_i;
end

yte = y_est(use_flag_ref_est) - sum(X_vary_est(use_flag_ref_est,:) .* At_vary_hat_est(use_flag_ref_est,:),2);
para_hat_ref = (XZ_const_nonzero_est(use_flag_ref_est,:)' * XZ_const_nonzero_est(use_flag_ref_est,:) + eye(size(XZ_const_nonzero_est,2)) * (1e-4))\(XZ_const_nonzero_est(use_flag_ref_est,:)' * yte);

At_const_nonzero_hat_test = para_hat_ref(1:length(const_id));
At_const_nonzero_hat_test = repmat(At_const_nonzero_hat_test',n_test,1);

At_hat_ref_test = zeros(n_test,R*S);
At_hat_ref_test(:, vary_id) = At_vary_hat_test;
At_hat_ref_test(:, const_id) = At_const_nonzero_hat_test;
At_hat_ref_test = reshape(At_hat_ref_test, n_test,R,S);

beta_nonzero_hat_test = para_hat_ref(end-length(beta_nonzero_id)+1:end);
beta_hat_ref_test = zeros(p0,1);
beta_hat_ref_test(beta_nonzero_id) = beta_nonzero_hat_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute estimated error and do refined regression
y_hat_ref_test = y_test;
y_hat_ref_test(use_flag_ref_test) = sum(X_test(use_flag_ref_test,:,:) .* At_hat_ref_test(use_flag_ref_test,:,:), [2,3]) + Z_test(use_flag_ref_test,:) * beta_hat_ref_test;
err_hat_ref_test = y_test - y_hat_ref_test;

err_hat_gt{i_n,1}(test_flag) = err_hat_ref_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B-spline projection
% here we use all time points in spline model
q = 4; n_knot = 6;
knots = linspace(0,1,n_knot);
Bst=bspline_basismatrix(q,[zeros(1,q-1), knots, ones(1,q-1)], t_train);
L=size(Bst,2);

%spline
Bs = reshape(Bst,[n_train,1,1,L]);
X_Bs = X_train .* Bs;
X_Bs = permute(X_Bs, [4,2,3,1]); % L-R-S-n
%tic;
[beta_hat_init,gamma_hat_cp_init,glmstats1] = kruskal_reg(Z_train,tensor(X_Bs),y_train,2,'normal');
% toc;
gamma_hat_init=double(full(gamma_hat_cp_init)); % L-R-S

At_hat_init = double(ttt(tensor(Bst), tensor(gamma_hat_init),2,1));  % n-R-S


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% different penalty type
for pentype = ["LASSO", "SCAD", "MCP"]
fprintf('For penalty: %s\n', pentype);
if strcmp(pentype, 'LASSO') 
penparam_beta=0.04; penparam_gamma_c=0.003; penparam_gamma_v=0.002;
elseif strcmp(pentype, 'SCAD')
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.07;
elseif strcmp(pentype, 'MCP')
penparam_beta=0.04; penparam_gamma_c=0.02; penparam_gamma_v=0.07;
else
error('no such penalty');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add penalty to distinguish const_0, const_not0, and varying coefficient
% in gamma and beta
% step 1: initialize gamma, beta and penalty weight 
gamma_old = gamma_hat_init;
beta_old = beta_hat_init;

dif=1;
iter=0;
while dif>dif_thrs && iter<tot_iter 
iter=iter+1;
% step2: given gamma, update beta
y_tilde = y_train - double(ttt(tensor(X_Bs), tensor(gamma_old), [1,2,3], [1,2,3])); % n-by-1
omega_beta = diag(dp(beta_old,penparam_beta,pentype)./(abs(beta_old)+1e-10));
left=Z_train' * Z_train + n_train/2 * omega_beta;
right=Z_train' * y_tilde;
beta_new = left \ right;

% step3: given beta, update gamma
% update by part, in each part, update gamma_rs=[gamma_1rs, gamma_2rs, ..,
% gamma_Lrs] L parameters. So here are total RS parts
gamma_new=gamma_old;
for s=1:S
gamma_s_old=squeeze(gamma_old(:,:,s));
y_tilde = y_train - double(ttt(tensor(X_Bs), tensor(gamma_new), [1,2,3], [1,2,3])) - Z_train * beta_new...
    + double(ttt(tensor(squeeze(X_Bs(:,:,s,:))), tensor(gamma_s_old), [1,2],[1,2])); % n-by-1
X_Bs_tilde = double(reshape(X_Bs(:,:,s,:), [L*R, n_train]))'; %n-by-LR
omega_gamma_c = zeros(L*R, L*R);
omega_gamma_v = zeros(L*R, L*R);
for r = 1:R
gamma_rs_old = squeeze(gamma_s_old(:,r));
omega_gamma_c((r-1)*L+1:r*L, (r-1)*L+1:r*L) = dp(abs(mean(gamma_rs_old)),penparam_gamma_c, pentype)...
    /(abs(mean(gamma_rs_old))+1e-10) * 1/(L*L)*ones(L,L);
omega_gamma_v((r-1)*L+1:r*L, (r-1)*L+1:r*L) = dp(norm(gamma_rs_old-mean(gamma_rs_old)),penparam_gamma_v, pentype)...
    / (norm(gamma_rs_old-mean(gamma_rs_old))+1e-10) * (eye(L)-1/L*ones(L,L));
end
left = (X_Bs_tilde')*X_Bs_tilde + n_train/2*omega_gamma_c + n_train/2*omega_gamma_v;
right = (X_Bs_tilde')*y_tilde;
gamma_s_new = left\right;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find out zero constant coefficient, non-zeros constant coefficient and
% varying coefficients.
% set hyper parameters
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

X_mat_train = reshape(X_train, n_train, R*S);  % n-by-R*S
X_mat_test = reshape(X_test, n_test, R*S);  % n-by-R*S

%constant coefficient
beta_nonzero_id = find(flag_beta_nonzero==1);
Z_nonzero_train = Z_train(:,beta_nonzero_id);
Z_nonzero_test = Z_test(:,beta_nonzero_id);

[const_row_idx, const_col_id] = find(flag_At_const_nonzero==1);
const_id = sub2ind(size(At_hat_pen,2:3), const_row_idx, const_col_id);
X_const_nonzero_train = X_mat_train(:,const_id);
X_const_nonzero_test = X_mat_test(:,const_id);

XZ_const_nonzero_train = [X_const_nonzero_train, Z_nonzero_train];
XZ_const_nonzero_test = [X_const_nonzero_test, Z_nonzero_test];

%varying coefficient
[vary_row_id, vary_col_id] = find(flag_At_vary==1);
vary_id = sub2ind(size(At_hat_pen,2:3), vary_row_id, vary_col_id);
X_vary_train = X_mat_train(:,vary_id);
X_vary_test = X_mat_test(:,vary_id);

At_vary_hat_test = zeros(n_test, length(vary_id));
use_flag_ref_test = true(n_test,1);
for i=1:n_test
ker = max(.75*(1-(t_train-t_test(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

if sum(ker > 0) <= 1.1*(2*length(vary_id)+length(const_id)+length(beta_nonzero_id))
    use_flag_ref_test(i) = false;
    continue
end

y_star = y_train .* ker;
XZ_const_nonzero_star = XZ_const_nonzero_train .* repmat(ker, 1, length(const_id)+length(beta_nonzero_id));

X_vary_a = X_vary_train;
sst = (t_train - t_test(i))/h2;
X_vary_b = X_vary_a .* repmat(sst, 1, length(vary_id));

X_vary_a = X_vary_a .* repmat(ker, 1, length(vary_id));
X_vary_b = X_vary_b .* repmat(ker, 1, length(vary_id));

XZ_star = [X_vary_a, X_vary_b, XZ_const_nonzero_star];
para_hat_ref = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_vary_hat_i = para_hat_ref(1:length(vary_id));
At_vary_hat_test(i,:) = At_vary_hat_i;
end

% estimation beta for Z without information of test set.
n_est = 0.1*n_train;
id_est = randperm(n_train, n_est)';
t_est = t_train(id_est);

y_est = y_train(id_est);
Z_nonzero_est = Z_nonzero_train(id_est, :);
X_const_nonzero_est = X_const_nonzero_train(id_est, :);
XZ_const_nonzero_est = [X_const_nonzero_est, Z_nonzero_est];
X_vary_est = X_vary_train(id_est,:);

At_vary_hat_est = zeros(n_est, length(vary_id));
use_flag_ref_est = true(n_est,1);
for i=1:n_est
ker = max(.75*(1-(t_train-t_est(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

if sum(ker > 0) <= 1.1*(2*length(vary_id)+length(const_id)+length(beta_nonzero_id))
    use_flag_ref_est(i) = false;
    continue
end

y_star = y_train .* ker;
XZ_const_nonzero_star = XZ_const_nonzero_train .* repmat(ker, 1, length(const_id)+length(beta_nonzero_id));

X_vary_a = X_vary_train;
sst = (t_train - t_est(i))/h2;
X_vary_b = X_vary_a .* repmat(sst, 1, length(vary_id));

X_vary_a = X_vary_a .* repmat(ker, 1, length(vary_id));
X_vary_b = X_vary_b .* repmat(ker, 1, length(vary_id));

XZ_star = [X_vary_a, X_vary_b, XZ_const_nonzero_star];
para_hat_ref = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_vary_hat_i = para_hat_ref(1:length(vary_id));
At_vary_hat_est(i,:) = At_vary_hat_i;
end

yte = y_est(use_flag_ref_est) - sum(X_vary_est(use_flag_ref_est,:) .* At_vary_hat_est(use_flag_ref_est,:),2);
para_hat_ref = (XZ_const_nonzero_est(use_flag_ref_est,:)' * XZ_const_nonzero_est(use_flag_ref_est,:) + eye(size(XZ_const_nonzero_est,2)) * (1e-4))\(XZ_const_nonzero_est(use_flag_ref_est,:)' * yte);

At_const_nonzero_hat_test = para_hat_ref(1:length(const_id));
At_const_nonzero_hat_test = repmat(At_const_nonzero_hat_test',n_test,1);

At_hat_ref_test = zeros(n_test,R*S);
At_hat_ref_test(:, vary_id) = At_vary_hat_test;
At_hat_ref_test(:, const_id) = At_const_nonzero_hat_test;
At_hat_ref_test = reshape(At_hat_ref_test, n_test,R,S);

beta_nonzero_hat_test = para_hat_ref(end-length(beta_nonzero_id)+1:end);
beta_hat_ref_test = zeros(p0,1);
beta_hat_ref_test(beta_nonzero_id) = beta_nonzero_hat_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute estimated error and do refined regression
y_hat_ref_test = y_test;
y_hat_ref_test(use_flag_ref_test) = sum(X_test(use_flag_ref_test,:,:) .* At_hat_ref_test(use_flag_ref_test,:,:), [2,3]) + Z_test(use_flag_ref_test,:) * beta_hat_ref_test;
err_hat_ref_test = y_test - y_hat_ref_test;

if strcmp(pentype, 'LASSO')
err_hat_lasso{i_n,1}(test_flag) = err_hat_ref_test;
elseif strcmp(pentype, 'SCAD')
err_hat_scad{i_n,1}(test_flag) = err_hat_ref_test;
elseif strcmp(pentype, 'MCP')
err_hat_mcp{i_n,1}(test_flag) = err_hat_ref_test;
else
error('no such penalty');
end

end
end
end

save('case4_pred_err.mat', 'err_hat_gt', 'err_hat_lasso', 'err_hat_scad', 'err_hat_mcp')


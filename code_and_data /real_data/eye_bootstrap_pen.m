% do the bootstrap on eye dataset to show the confidence interval of
% estimated parameters.

% semi-parametric model for eye dataset

clear all;

load('GRAPE_vector.mat');
age = vct_covar.Age;

load('covariate_pen_cfp.mat')
% load('covariate_pen_roi.mat')

p0 = size(Z, 2);
n = length(t);

S = size(X,3);
R = size(X,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after identify the constant coefficients and varying coefficients
% do a refine regression 
X_mat = reshape(X, n, []);

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% coefficient structure for CFP
beta_nonzero_id = [2,3,6];
const_id = [2];

%varying coefficient
vary_id = [1,3,4,5,6,11,12,13,15,16,17,18];
%}

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% coefficient structure for ROI
%constant coefficient
beta_nonzero_id = [2,3,6];
const_id = [2,14,18];

%varying coefficient
vary_id = [1,3,4,5,6,7,8,11,12,13,15,16,17];
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%constant coefficient
beta_nonzero_id = [2,3,6];
const_id = [2];

%varying coefficient
vary_id = [1,3,4,5,6,11,12,13,15,16,17,18];

n_rep = 500;
n_btsp = n;
A_hat_btsp = zeros(n_rep, n, R*S);
A_use_flag_btsp = zeros(n_rep, n);
beta_hat_btsp = zeros(n_rep, p0);
rmse_btsp = zeros(1, n_rep);

for i_rep=1:n_rep
btsp_id = randi(n, [1,n_btsp]);
% btsp_id = 1:n;

Z_btsp = Z(btsp_id,:);
y_btsp = y(btsp_id,:);
X_mat_btsp = X_mat(btsp_id,:);
t_btsp = t(btsp_id);

Z_btsp_nonzero = Z_btsp(:,beta_nonzero_id);
X_btsp_const_nonzero = X_mat_btsp(:,const_id);
const_nonzero_btsp = [X_btsp_const_nonzero, Z_btsp_nonzero];
X_btsp_vary = X_mat_btsp(:,vary_id);

A_hat_vary = zeros(n, length(vary_id));
A_unuse_id = [];

h2=0.2;
for i = 1:n
ker = max(.75*(1-(t_btsp-t(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

if sum(ker ~= 0) < 1.2 * (2*R*S+p0)
    A_unuse_id = [A_unuse_id,i];
    continue
end

y_star = y_btsp .* ker;
const_nonzero_star = const_nonzero_btsp .* repmat(ker,1,size(const_nonzero_btsp,2));
% Z_ref_nonzero_star = Z_ref_nonzero .* repmat(ker,1,size(Z_ref_nonzero,2));

X_a = X_btsp_vary;
sst = (t_btsp-t(i))/h2;
X_b = X_a .* repmat(sst, 1, length(vary_id));

X_a = X_a .* repmat(ker, 1, length(vary_id));
X_b = X_b .* repmat(ker, 1, length(vary_id));

XZ_star = [X_a, X_b, const_nonzero_star];

% n_row_all0 = sum(all(XZ_star == 0, 2));
para_hat = (XZ_star' * XZ_star + 1e-4) \ XZ_star' * y_star;
A_hat_i = para_hat(1:length(vary_id));
A_hat_vary(i,:) = A_hat_i; 
end

A_unuse_flag = ismember(1:n, A_unuse_id);
A_use_flag = ~A_unuse_flag;
% sum(A_use_flag)

A_btsp_flag = A_use_flag(btsp_id);

yte = y_btsp(A_btsp_flag) - sum(X_btsp_vary(A_btsp_flag, :) .* A_hat_vary(A_btsp_flag, :), 2);
para_hat_const = (const_nonzero_btsp(A_btsp_flag, :)'*const_nonzero_btsp(A_btsp_flag, :) + eye(size(const_nonzero_btsp,2))*(1e-4))\(const_nonzero_btsp(A_btsp_flag, :)'*yte);

beta_hat_nonzero = para_hat_const(end-length(beta_nonzero_id)+1:end);
beta_hat = zeros(p0,1);
beta_hat(beta_nonzero_id) = beta_hat_nonzero;

A_hat_const_nonzero = para_hat_const(1:length(const_id));
A_hat_const_nonzero = repmat(A_hat_const_nonzero',n,1);

A_hat = zeros(n,R*S);
A_hat(:, vary_id) = A_hat_vary;
A_hat(:, const_id) = A_hat_const_nonzero;

y_hat = sum(X_mat_btsp(A_btsp_flag,:) .* A_hat(A_btsp_flag, :), 2) + Z_btsp(A_btsp_flag, :) * beta_hat;
err_hat = y_btsp - y_hat;

rmse = sqrt(mean(err_hat.^2, 'all'));

A_hat_btsp(i_rep, :, :, :) = A_hat;
A_use_flag_btsp(i_rep, :) = A_use_flag;
beta_hat_btsp(i_rep, :) = beta_hat;
rmse_btsp(i_rep) = rmse;

fprintf('This is %d-th repetition, RMSE = %.4f \n', i_rep, rmse)
end

% save('eye_cfp_pen_btsp500_R2_S9.mat', 'rmse_btsp', 'A_hat_btsp', 'A_use_flag_btsp', 'beta_hat_btsp', 'X_part', 't', 'n_rep');
% save('eye_roi_pen_btsp500_R2_S9.mat', 'rmse_btsp', 'A_hat_btsp', 'A_use_flag_btsp', 'beta_hat_btsp', 'X_part', 't', 'n_rep');


%{
btsp_res = load('eye_roi_pen_btsp500_R2_S9.mat');
% btsp_res = load('eye_cfp_pen_btsp500_R2_S9.mat');
A_hat_btsp = btsp_res.A_hat_btsp;
beta_hat_btsp = btsp_res.beta_hat_btsp;
X_part = btsp_res.X_part;
t = btsp_res.t;
n_rep = btsp_res.n_rep;
%}


t_plot = t;
r_ls = [1,2];
s_ls = [1,2,3,4,5,6,7,8,9];
figure('Position', [100, 100, 1300, 400]);
tiledlayout(3,6,'Padding','compact','TileSpacing','compact'); 
for s_id=1:length(s_ls)
for r_id=1:length(r_ls)
s = s_ls(s_id);
r = r_ls(r_id);
idx = (s_id-1)*2+r_id;
a_rs = squeeze(A_hat_btsp(:,:,(s-1)*2+r));
mean_a_plot=mean(a_rs,1);
pt5_a_plot=prctile(a_rs,5, 1);
pt95_a_plot=prctile(a_rs, 95, 1);
nexttile;
hold on;
plot(t_plot, mean_a_plot, '-', 'LineWidth', 2);
plot(t_plot, pt5_a_plot, '--', 'LineWidth', 2);
plot(t_plot, pt95_a_plot, '--', 'LineWidth', 2);
xlim([18, 80]);
ylim([-1.5, 1.5]);
hold off;
ylabel(['a_{', num2str(r_plot), ',', num2str(s_plot), '}(t)']);
end
end

mean_b=mean(beta_hat_btsp,1);
pt5_b=prctile(beta_hat_btsp,5,1);
pt95_b=prctile(beta_hat_btsp,95,1);



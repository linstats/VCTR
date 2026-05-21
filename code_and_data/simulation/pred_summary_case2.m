% This is the code for the prediction error of summarying 3 different
% models in case 2:
% 1. our proposed VCTR model; 
% 2. tensor partition model: Miranda et al. (2018);
% 3. tensor kruskal regression model: Zhou et al. (2013).

clear all;

% varying coefficient index
n = 2000;    % sample size
% n = 5000;
n_At = 4;

D=3;
p1=40; p2=40; p3=40;
S1=4;S2=4;S3=4;
p1_s = p1/S1; p2_s = p2/S2; p3_s = p3/S3;
R=5;
S=S1*S2*S3;

p0 = 2;

h1 = 0.2;
t = sort(rand(n,1)); % index variable

k_fold = 10;
CVP=cvpartition(n,'kFold',k_fold); %% 10-fold cv

% the flag of whether using sample i in vcplt model 
use_flag = false(n,1);
for i_fold = 1:k_fold
n_test = CVP.TestSize(i_fold);

train_flag = training(CVP,i_fold);
test_flag=test(CVP,i_fold);

t_train = t(train_flag);
t_test = t(test_flag);

use_flag_test = true(n_test,1);
for i=1:n_test
ker = 1-(t_train-t_test(i)).^2/(h1^2);
if sum(ker > 0) <= 1.1*(2*R*S+p0)
    use_flag_test(i) = false;
end
end

use_flag(test_flag) = use_flag_test;
end

% ground truth for 4 A(t)'s
err_gt = zeros(n,n_At);

% prediction results of vcplt model
At_hat_vcplt = zeros(n, R, S, n_At);
err_hat_vcplt = zeros(n, n_At);

% prediction results of tensor partition model
At_hat_ptt = zeros(n, R, S, n_At);
err_hat_ptt =zeros(n, n_At);

% prediction results of tensor kruskal model
At_cal_hat_krus = zeros(n, p1, p2, p3,n_At);
At_hat_krus = zeros(n, R, S, n_At);
err_hat_krus =zeros(n, n_At);

for i_At = 1:n_At 
fprintf('For A_%d(t):\n', i_At);
% 2D varying coefficient
bR = sqrt((1:R)'/R);
bS = sqrt((1:S)'/S);
if i_At == 1
    At = full(ktensor({4*(t-0.5).^2,bR, bS}));
elseif i_At == 2
    At = full(ktensor({t.^0.5,bR, bS}));
elseif i_At == 3
    At = full(ktensor({1.75*((exp(-(3*t-1).^2)+exp(-(4*t-3).^2))-0.75),bR, bS}));
else
    At = full(ktensor({sin(2*pi*(t-0.5)),bR, bS}));
end
At = double(At);

X = randn(n,R,S);

X1 = zeros(p1_s,R,S);
X2 = zeros(p2_s,R,S);
X3 = zeros(p3_s,R,S);
for s = 1:S
X1_s = randn(p1_s,R); [X1_s, ~] = qr(X1_s,0); X1(:,:,s) = X1_s; 
X2_s = randn(p2_s,R); [X2_s, ~] = qr(X2_s,0); X2(:,:,s) = X2_s;
X3_s = randn(p3_s,R); [X3_s, ~] = qr(X3_s,0); X3(:,:,s) = X3_s;
end

lambda = ones(R,1);

At_cal = zeros(n,p1,p2,p3);
X_cal = zeros(n,p1,p2,p3);
for s1 = 1:S1
for s2 = 1:S2
for s3 = 1:S3
s = (s1-1)*S2*S3+(s2-1)*S3+s3;
At_s = At(:,:,s);
X_s = X(:,:,s);
X1_s = X1(:,:,s);
X2_s = X2(:,:,s);
X3_s = X3(:,:,s);
At_cal(:, (s1-1)*p1_s+1:s1*p1_s, (s2-1)*p2_s+1:s2*p2_s, (s3-1)*p3_s+1:s3*p3_s) = ktensor(lambda, At_s, X1_s, X2_s, X3_s);
X_cal(:, (s1-1)*p1_s+1:s1*p1_s, (s2-1)*p2_s+1:s2*p2_s, (s3-1)*p3_s+1:s3*p3_s) = ktensor(lambda, X_s, X1_s, X2_s, X3_s);
end
end
end

% True coefficients for regular (non-array) covariates
beta = 3*ones(p0,1);

Z = randn(n,p0);   % n-by-p0 regular design matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = sum(X_cal.*At_cal,[2,3,4]) + Z * beta;
% mu = sum(X .* At, [2,3]) + Z * beta;

err = randn(n,1);
err_gt(:, i_At) = err;

y = mu + err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use 3 models to do the prediction, 
% we predict each sample \hat{y}_i and compare with true value y_i to
% compute prediction error
for i_fold = 1:k_fold
fprintf('This is %d-th fold.\n', i_fold);
n_train = CVP.TrainSize(i_fold);
n_test = CVP.TestSize(i_fold);

train_flag = training(CVP,i_fold);
test_flag=test(CVP,i_fold);

t_train = t(train_flag);
t_test = t(test_flag);

y_train=y(train_flag);
y_test=y(test_flag);

X_cal_train = X_cal(train_flag,:,:,:);
X_cal_test = X_cal(test_flag,:,:,:);

X_train = X(train_flag,:,:);
X_test = X(test_flag,:,:);

Z_train = Z(train_flag,:);
Z_test = Z(test_flag,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first model: our VCTR model

tic;
At_hat_vcplt_test = zeros(n_test, R, S);
use_flag_test = use_flag(test_flag);
for i=1:n_test
if use_flag_test(i) == false
    continue
end

ker = max(.75*(1-(t_train-t_test(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

y_star = y_train .* ker;
Z_star = Z_train .* repmat(ker, 1, p0);
X_mat = reshape(X_train, n_train, R*S);
X_mat_a = X_mat;
sst = (t_train - t_test(i))/h1;
X_mat_b = X_mat_a .* repmat(sst, 1, R*S);

X_mat_a = X_mat_a .* repmat(ker, 1, R*S);
X_mat_b = X_mat_b .* repmat(ker, 1, R*S);

XZ_star = [X_mat_a, X_mat_b, Z_star];
para_hat = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_hat_i = para_hat(1:R*S);
At_hat_vcplt_test(i,:,:) = reshape(At_hat_i, R,S);
end
At_hat_vcplt(test_flag,:,:, i_At) = At_hat_vcplt_test;

% estimation beta for Z without information of test set.
n_est = 0.5*n_train;
id_est = randperm(n_train, n_est)';
t_est = t_train(id_est);

y_est = y_train(id_est);
X_est = X_train(id_est, :, :);
Z_est = Z_train(id_est, :);

At_hat_vcplt_est = zeros(n_est, R, S);
use_flag_est = true(n_est,1);
for i=1:n_est
ker = max(.75*(1-(t_train-t_est(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker > 0) <= 1.1*(2*R*S+p0)
    use_flag_est(i) = false;
    continue
end

y_star = y_train .* ker;
Z_star = Z_train .* repmat(ker, 1, p0);
X_mat = reshape(X_train, n_train, R*S);
X_mat_a = X_mat;
sst = (t_train - t_est(i))/h1;
X_mat_b = X_mat_a .* repmat(sst, 1, R*S);

X_mat_a = X_mat_a .* repmat(ker, 1, R*S);
X_mat_b = X_mat_b .* repmat(ker, 1, R*S);

XZ_star = [X_mat_a, X_mat_b, Z_star];
para_hat = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_hat_i = para_hat(1:R*S);
At_hat_vcplt_est(i,:,:) = reshape(At_hat_i, R,S);
end

yte = y_est(use_flag_est) - sum(X_est(use_flag_est,:,:) .* At_hat_vcplt_est(use_flag_est,:,:), [2,3]);
beta_hat_vcplt_test = (Z_est(use_flag_est, :)'*Z_est(use_flag_est, :) + eye(p0)*(1e-4))\(Z_est(use_flag_est, :)'*yte);

y_hat_vcplt_test = y_test;
y_hat_vcplt_test(use_flag_test) = sum(X_test(use_flag_test,:,:) .* At_hat_vcplt_test(use_flag_test,:,:), [2,3]) + Z_test(use_flag_test,:) * beta_hat_vcplt_test;
err_hat_vcplt_test = y_test - y_hat_vcplt_test;

err_hat_vcplt(test_flag, i_At) = err_hat_vcplt_test;
toc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% second model: tensor partition model - Miranda et al. (2018)

X_mat = reshape(X_train, n_train, R*S);
XZ_train = [X_mat, Z_train];

para_hat = (XZ_train' * XZ_train) \ (XZ_train' * y_train);
At_hat_ptt_test = repmat(para_hat(1:R*S)', n_test,1);
At_hat_ptt_test = reshape(At_hat_ptt_test, n_test, R, S);
beta_hat_ptt_test = para_hat(R*S+1:end);

At_hat_ptt(test_flag, :, :, i_At) = At_hat_ptt_test;

y_hat_ptt_test = sum(X_test .* At_hat_ptt_test, [2,3]) + Z_test * beta_hat_ptt_test;
err_hat_ptt_test = y_test - y_hat_ptt_test;

err_hat_ptt(test_flag, i_At) = err_hat_ptt_test;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% third model: kruskal regression model - Zhou et al. (2013)
X_krus_train = tensor(permute(X_cal_train, [2,3,4,1]));

tic;
[beta_hat_krus_test,At_cal_hat_krus_test,glmstats] = kruskal_reg(Z_train,X_krus_train,y_train,1,'normal','MaxIter', 100,'Replicates', 1);
toc;
At_cal_hat_krus_test=double(full(At_cal_hat_krus_test)); % R-S
At_cal_hat_krus_test = reshape(At_cal_hat_krus_test, [1,p1,p2,p3]);
At_cal_hat_krus_test = repmat(At_cal_hat_krus_test, [n_test,1,1]);

At_cal_hat_krus(test_flag,:,:,:,i_At) = At_cal_hat_krus_test;

y_hat_krus_test = sum(X_cal_test .* At_cal_hat_krus_test, [2,3,4]) + Z_test * beta_hat_krus_test;
err_hat_krus_test = y_test - y_hat_krus_test;

err_hat_krus(test_flag,i_At) = err_hat_krus_test;

% project to At 
At_hat_krus_test = zeros(n_test,R,S);
for s1 = 1:S1
for s2 = 1:S2
for s3 = 1:S3
s = (s1-1)*S2*S3+(s2-1)*S3+s3;
X1_s = X1(:,:,s);
X2_s = X2(:,:,s);
X3_s = X3(:,:,s);
aux_mat = diag(lambda) * khatrirao(X3_s, X2_s, X1_s)';
At_cal_hat_krus_test_s = At_cal_hat_krus_test(:, (s1-1)*p1_s+1:s1*p1_s, (s2-1)*p2_s+1:s2*p2_s, (s3-1)*p3_s+1:s3*p3_s);
At_hat_krus_test(:,:,s) = reshape(At_cal_hat_krus_test_s, n_test, []) / aux_mat;
end
end
end
At_hat_krus(test_flag,:,:,i_At) = At_hat_krus_test;

end
end

%{
i_At = 4;
boxplot([err_gt(use_flag, i_At), err_hat_vcplt(use_flag, i_At), err_hat_ptt(use_flag, i_At), err_hat_krus(use_flag, i_At)])
%}


save('case2_n5000.mat', 'use_flag', 'err_gt', 'At_hat_vcplt', 'err_hat_vcplt', 'At_hat_ptt', 'err_hat_ptt', 'At_hat_krus', 'err_hat_krus')




%{
mae_At_vcplt = mean(abs(At(use_flag,:,:) - At_hat_vcplt(use_flag,:,:,1)),'all');
rmse_At_vcplt = sqrt(mean((At(use_flag,:,:) - At_hat_vcplt(use_flag,:,:,1)).^2,'all'));
mae_err_vcplt = mean(abs(err_hat_vcplt(use_flag,1)),'all');
rmse_err_vcplt = sqrt(mean(err_hat_vcplt(use_flag,1).^2, 'all'));
fprintf('In VCPLT model, for At: MAE is %.4f, RMSE is %.4f, for error: MAE is %.4f, RMSE is %.4f\n', ...
    mae_At_vcplt, rmse_At_vcplt, mae_err_vcplt, rmse_err_vcplt);

mae_At_ptt = mean(abs(At(use_flag,:,:) - At_hat_ptt(use_flag,:,:,1)),'all');
rmse_At_ptt = sqrt(mean((At(use_flag,:,:) - At_hat_ptt(use_flag,:,:,1)).^2,'all'));
mae_err_ptt = mean(abs(err_hat_ptt(use_flag,1)),'all');
rmse_err_ptt = sqrt(mean(err_hat_ptt(use_flag,1).^2, 'all'));
fprintf('In constant tensor partition model, for At: MAE is %.4f, RMSE is %.4f, for error: MAE is %.4f, RMSE is %.4f\n', ...
    mae_At_ptt, rmse_At_ptt, mae_err_ptt, rmse_err_ptt);

mae_At_krus = mean(abs(At(use_flag,:,:) - At_hat_krus(use_flag,:,:,1)),'all');
rmse_At_krus = sqrt(mean((At(use_flag,:,:) - At_hat_krus(use_flag,:,:,1)).^2,'all'));
mae_At_cal_krus = mean(abs(At_cal(use_flag,:,:,:) - At_cal_hat_krus(use_flag,:,:,:,1)),'all');
rmse_At_cal_krus = sqrt(mean((At_cal(use_flag,:,:,:) - At_cal_hat_krus(use_flag,:,:,:,1)).^2,'all'));
mae_err_krus = mean(abs(err_hat_krus(use_flag,1)),'all');
rmse_err_krus = sqrt(mean(err_hat_krus(use_flag,1).^2, 'all'));
fprintf('In tensor kruskal model, for At: MAE is %.4f, RMSE is %.4f, for At_cal: MAE is %.4f, RMSE is %.4f, for error: MAE is %.4f, RMSE is %.4f\n', ...
    mae_At_krus, rmse_At_krus, mae_At_cal_krus, rmse_At_cal_krus, mae_err_krus, rmse_err_krus);
%}


% semi-parametric model for eye dataset

clear all;
%--------------------------------
%   Prepare eye image X, y and z
%--------------------------------

% load y t and z
load('GRAPE_vector.mat');

t = vct_covar.Age;
% histogram(t)
t = (t - min(t))/(max(t) - min(t));
% n = length(t);

t_thres = 0.8;
flt_flag = t < t_thres;
t = t(flt_flag);
t = (t - min(t))/(max(t) - min(t));
n = sum(flt_flag);

% y = vct_covar.IOP;
y = vct_covar.IOP(flt_flag);
y = zscore(y);

p0 = 6;
Z = zeros(n, p0);
Z(:,1) = vct_covar.IsFemale(flt_flag);
Z(:,2) = vct_covar.VF1(flt_flag);
Z(:,3) = vct_covar.VF2(flt_flag);
Z(:,4) = vct_covar.VF3(flt_flag);
Z(:,5) = vct_covar.VF4(flt_flag);
Z(:,6) = vct_covar.VF5(flt_flag);
Z(:,2:6) = zscore(Z(:,2:6));

[t, sort_idx] = sort(t);
y = y(sort_idx);
Z = Z(sort_idx,:);

% load image tensor X
% image_path = '/Users/hejiaxin/Documents/MATLAB/tensor_semipara/realdata_eye/eye_dataset/ROI_images';
image_path = '/Users/hejiaxin/Documents/MATLAB/tensor_semipara/realdata_eye/eye_dataset/CFPs';

%load eye images according to the covariate table. If it is a OS image,
%flip it. 
%image_name = vct_covar.CFP;
image_name = vct_covar.CFP(flt_flag);
image_name = image_name(sort_idx);

%{
% load every fundus image and we need normalize them. 
% try to load one fundus image and do the grayscale transformation
img_path = fullfile(image_path, image_name{7});
eye_img = imread(img_path);
eye_img = imresize3(eye_img, [p1,p2,p3]);
% imshow(eye_img)
eye_img_gray = rgb2gray(eye_img);
% imshow(eye_img_gray)

subplot(1, 2, 1);
imshow(eye_img);
title('original fundus image');

subplot(1, 2, 2);
imshow(eye_img_gray);
title('grayscale fundus image');
%}

p1=192; p2=192; p3=3;
X_cal = zeros(n,p1,p2,p3);
for i = 1:n
    img_path = fullfile(image_path, image_name{i});
    eye_img = imread(img_path);
    
    if contains(image_name{i}, 'OS')
        eye_img = fliplr(eye_img);
    end
    
    img = imresize3(eye_img, [p1,p2,p3]);
    X_cal(i,:,:,:) = img;
    clear eye_img img;
end

% divide it as S1=4 S2=4, S3=1
S1 = 3; S2 = 3; S3 = 1; S = S1*S2*S3;
R = 2;

X_part = cell(S,2);
for s1 = 1:S1
for s2 = 1:S2
s = (s2 - 1) * S1 + s1;
X_cal_s = X_cal(:, (s1-1)*(p1/S1)+1:s1*(p1/S1), (s2-1)*(p2/S2)+1:s2*(p2/S2), :);
X_cal_s_cp = cp_als(tensor(X_cal_s), R, 'printitn', 0);
X_cal_s_cp_lambda = X_cal_s_cp.lambda;
X_cal_s_cp_u1 = X_cal_s_cp.U{1};
X_cal_s_cp_u2 = X_cal_s_cp.U{2};
X_cal_s_cp_u3 = X_cal_s_cp.U{3};
X_cal_s_cp_u4 = X_cal_s_cp.U{4};
X_cal_s_cp_u1_sd = std(X_cal_s_cp_u1,1);
X_cal_s_cp_u1 = X_cal_s_cp_u1 ./ X_cal_s_cp_u1_sd;
X_cal_s_cp_lambda = X_cal_s_cp_lambda .* X_cal_s_cp_u1_sd';
X_cal_s_SV = zeros(R, p1/S1*p2/S2*p3/S3);
for r =1:R
X_cal_s_SV(r,:) = reshape(double(ktensor(X_cal_s_cp_lambda(r), X_cal_s_cp_u2(:,r), X_cal_s_cp_u3(:,r), X_cal_s_cp_u4(:,r))), p1/S1*p2/S2*p3/S3, 1);
end
X_part{s,1} = X_cal_s_cp_u1;
X_part{s,2} = X_cal_s_SV;
end
end

X = zeros(n, R, S);
for s1 = 1:S1
for s2 = 1:S2
    s = (s2 - 1) * S1 + s1;
    X_s = X_part{s,1};
    X(:, :, s) = X_s;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Least square estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use kernel smoothing to do the prediction, 
% we predict each sample \hat{y}_i and compare with true value y_i to
% compute prediction error
k_fold = 10;
CVP=cvpartition(n,'kFold',k_fold); %% 10-fold cv

use_flag = false(n,1);
At_hat = zeros(n, R, S);
y_hat = zeros(n,1);
err_hat=zeros(n,1);

for i_fold = 1:k_fold
fprintf('This is %d-th fold.\n', i_fold);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VCPLT model
h1 = 0.13;

At_hat_test = zeros(n_test, R, S);
At_test_use_flag = true(n_test,1);
for i=1:n_test
ker = max(.75*(1-(t_train-t_test(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker > 0) <= 1.2 * (2*R*S+p0)
    At_test_use_flag(i) = false;
    continue
end

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
At_hat_test(i,:,:) = reshape(At_hat_i, R,S);
end
At_hat(test_flag,:,:) = At_hat_test;

% estimation beta for Z without information of test set.
At_hat_train = zeros(n_train, R, S);
At_train_use_flag = true(n_train,1);
for i=1:n_train
ker = max(.75*(1-(t_train-t_train(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker > 0) <= 1.2 * (2*R*S+p0)
    At_train_use_flag(i) = false;
    continue
end

y_star = y_train .* ker;
Z_star = Z_train .* repmat(ker, 1, p0);
X_mat = reshape(X_train, n_train, R*S);
X_mat_a = X_mat;
sst = (t_train - t_train(i))/h1;
X_mat_b = X_mat_a .* repmat(sst, 1, R*S);

X_mat_a = X_mat_a .* repmat(ker, 1, R*S);
X_mat_b = X_mat_b .* repmat(ker, 1, R*S);

XZ_star = [X_mat_a, X_mat_b, Z_star];
para_hat = (XZ_star' * XZ_star) \ XZ_star' * y_star;
At_hat_i = para_hat(1:R*S);
At_hat_train(i,:,:) = reshape(At_hat_i, R,S);
end

yte = y_train(At_train_use_flag) - sum(X_train(At_train_use_flag,:,:) .* At_hat_train(At_train_use_flag,:,:), [2,3]);
beta_hat = (Z_train(At_train_use_flag, :)'*Z_train(At_train_use_flag, :) + eye(p0)*(1e-4))\(Z_train(At_train_use_flag, :)'*yte);

y_hat_test = y_test;
y_hat_test(At_test_use_flag) = sum(X_test(At_test_use_flag,:,:) .* At_hat_test(At_test_use_flag,:,:), [2,3]) + Z_test(At_test_use_flag,:) * beta_hat;
err_hat_test = y_test - y_hat_test;

y_hat(test_flag) = y_hat_test;
err_hat(test_flag) = err_hat_test;
use_flag(test_flag) = At_test_use_flag;
end

mean(abs(err_hat(use_flag)),'all')
sqrt(mean(err_hat(use_flag).^2, 'all'))
mean((err_hat(use_flag)-mean(err_hat(use_flag))).^2)
mean(abs(err_hat),'all')
sqrt(mean(err_hat.^2, 'all'))
mean((err_hat-mean(err_hat)).^2)


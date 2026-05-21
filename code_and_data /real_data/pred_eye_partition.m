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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%, 
% we predict each sample \hat{y}_i and compare with true value y_i to
% compute prediction error
k_fold = 10;
CVP=cvpartition(n,'kFold',k_fold); %% 10-fold cv

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
% Tensor partition model
X_mat = reshape(X_train, n_train, R*S);
XZ_train = [X_mat, Z_train];

para_hat = (XZ_train' * XZ_train) \ (XZ_train' * y_train);
At_hat_test = repmat(para_hat(1:R*S)', n_test,1);
At_hat_test = reshape(At_hat_test, n_test, R, S);
beta_hat_test = para_hat(R*S+1:end);

At_hat(test_flag,:,:) = At_hat_test;

y_hat_test = sum(X_test .* At_hat_test, [2,3]) + Z_test * beta_hat_test;
err_hat_test = y_test - y_hat_test;

y_hat(test_flag) = y_hat_test;
err_hat(test_flag) = err_hat_test;
end

mean(abs(err_hat),'all')
sqrt(mean(err_hat.^2, 'all'))
mean((err_hat-mean(err_hat)).^2)

%{
figure;
hold on; 
plot(t, y);
plot(t, y_hat);
hold off;

figure;
hold on; 
plot(t, err_hat);
hold off;

%}

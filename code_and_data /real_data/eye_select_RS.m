% select R and S for semi-parametric model on eye dataset

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
%{
Z(:,1) = vct_covar.IsFemale;
Z(:,2) = vct_covar.VF1;
Z(:,3) = vct_covar.VF2;
Z(:,4) = vct_covar.VF3;
Z(:,5) = vct_covar.VF4;
Z(:,6) = vct_covar.VF5;
%}
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
img_path = fullfile(image_path, image_name{10});
eye_img = imread(img_path);
eye_img = imresize3(eye_img, [p1,p2,p3]);
imshow(eye_img)
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
S1 = 6; S2 = 6; S3 = 1; S = S1*S2*S3;
R_ls = 1:4;
k_fold = 10;
n_rep = 10;
mae = zeros(length(R_ls), n_rep, k_fold);
rmse = zeros(length(R_ls), n_rep, k_fold);

for i_R = 1:length(R_ls)
R = R_ls(i_R);
for i_rep = 1:n_rep

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

X_mat = reshape(X, n, []);
p_X_mat = size(X_mat,2); 

CVP=cvpartition(n,'kFold',k_fold); %% 10-fold cv

for i_fold = 1:k_fold
n_train = CVP.TrainSize(i_fold);
n_test = CVP.TestSize(i_fold);

train_flag = training(CVP,i_fold);
test_flag=test(CVP,i_fold);

t_train = t(train_flag);
t_test = t(test_flag);

y_train=y(train_flag,:);
y_test=y(test_flag,:);

X_mat_train = X_mat(train_flag,:,:);
X_mat_test = X_mat(test_flag,:,:);

Z_train = Z(train_flag,:);
Z_test = Z(test_flag,:);

% do the tensor varying coefficient regression 
A_hat_sm = zeros(n,p_X_mat);
A_unuse_id = [];
h1 = 0.2;
for i=1:n
ker = max(.75*(1-(t_train-t(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker ~= 0) < 1.2 * (2*p_X_mat+p0)
    A_unuse_id = [A_unuse_id,i];
    continue
end

y_star = y_train .* ker;
Z_star = Z_train .* repmat(ker,1,p0);
X_mat_a = X_mat_train;
sst = (t_train-t(i))/h1;
X_mat_b = X_mat_a .* repmat(sst, 1, p_X_mat);

X_mat_a = X_mat_a .* repmat(ker, 1, p_X_mat);
X_mat_b = X_mat_b .* repmat(ker, 1, p_X_mat);

XZ_star = [X_mat_a, X_mat_b, Z_star];
% n_row_all0 = sum(all(XZ_star == 0, 2));
para_hat_sm = (XZ_star' * XZ_star + 1e-4) \ XZ_star' * y_star;
A_hat_i = para_hat_sm(1:p_X_mat);
A_hat_sm(i,:) = A_hat_i;
beta_hat_i = para_hat_sm(2*p_X_mat+1:end);
end

A_unuse_flag = ismember(1:n, A_unuse_id)';
A_use_flag = ~A_unuse_flag;
% sum(A_use_flag)

A_train_flag = A_use_flag & train_flag;
A_test_flag = A_use_flag & test_flag;

yte = y(A_train_flag) - sum(X_mat(A_train_flag,:) .* A_hat_sm(A_train_flag, :), 2);
beta_hat_sm = (Z(A_train_flag, :)'*Z(A_train_flag, :) + eye(p0)*(1e-4))\(Z(A_train_flag, :)'*yte);

% s = 5; squeeze(A_hat_sm(:,s,:))'

y_hat_sm = sum(X_mat(A_test_flag,:) .* A_hat_sm(A_test_flag,:), 2) + Z(A_test_flag,:) * beta_hat_sm;
err_hat_sm = y(A_test_flag) - y_hat_sm;

mae(i_R, i_rep, i_fold) = mean(abs(err_hat_sm),'all');
rmse(i_R, i_rep, i_fold) = sqrt(mean(err_hat_sm.^2, 'all'));

end
fprintf('S=%d, R=%d, in %d-th repeat, in 10-fold cv, mae=%f4, rmse=%f4.\n', S, R, i_rep, squeeze(mean(mae(i_R, i_rep))), squeeze(mean(rmse(i_R, i_rep))))
end
end

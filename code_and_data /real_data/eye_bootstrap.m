% do the bootstrap on eye dataset to show the confidence interval of
% estimated parameters.

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

X_mat = reshape(X, n, []);
p_X_mat = size(X_mat,2); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do bootstrap and repeat 500 times
n_rep = 500;
n_btsp = n;
A_hat_btsp = zeros(n_rep, n, p_X_mat);
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

A_hat_sm = zeros(n,p_X_mat);
A_unuse_id = [];
h1 = 0.2;
for i=1:n
ker = max(.75*(1-(t_btsp-t(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker ~= 0) < 1.2 * (2*p_X_mat+p0)
    A_unuse_id = [A_unuse_id,i];
    continue
end

y_star = y_btsp .* ker;
Z_star = Z_btsp .* repmat(ker,1,p0);
X_mat_a = X_mat_btsp;
sst = (t_btsp-t(i))/h1;
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

A_unuse_flag = ismember(1:n, A_unuse_id);
A_use_flag = ~A_unuse_flag;
% sum(A_use_flag)

A_btsp_flag = A_use_flag(btsp_id);

yte = y_btsp(A_btsp_flag) - sum(X_mat_btsp(A_btsp_flag,:) .* A_hat_sm(A_btsp_flag, :), 2);
beta_hat_sm = (Z_btsp(A_btsp_flag, :)'*Z_btsp(A_btsp_flag, :) + eye(p0)*(1e-4))\(Z_btsp(A_btsp_flag, :)'*yte);

y_hat_sm = sum(X_mat_btsp(A_btsp_flag,:) .* A_hat_sm(A_btsp_flag, :), 2) + Z_btsp(A_btsp_flag, :) * beta_hat_sm;
err_hat_sm = y_btsp - y_hat_sm;

rmse = sqrt(mean(err_hat_sm.^2, 'all'));

A_hat_btsp(i_rep, :, :, :) = A_hat_sm;
A_use_flag_btsp(i_rep, :) = A_use_flag;
beta_hat_btsp(i_rep, :) = beta_hat_sm;
rmse_btsp(i_rep) = rmse;

fprintf('This is %d-th repetition, RMSE = %.4f \n', i_rep, rmse)
end

save('eye_btsp500_R2_S9.mat', 'rmse_btsp', 'A_hat_btsp', 'A_use_flag_btsp', 'beta_hat_btsp', 'X_part', 'n_rep');

%{
btsp_res = load('design2_btsp500_R3_S64.mat');
A_hat_btsp = btsp_res.A_hat_btsp;
beta_hat_btsp = btsp_res.beta_hat_btsp;
X_part = btsp_res.X_part;
n_rep = btsp_res.n_rep;
%}

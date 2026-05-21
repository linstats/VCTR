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
image_path = '/Users/hejiaxin/Documents/MATLAB/tensor_semipara/realdata_eye/eye_dataset/ROI_images';
% image_path = '/Users/hejiaxin/Documents/MATLAB/tensor_semipara/realdata_eye/eye_dataset/CFPs';

%load eye images according to the covariate table. If it is a OS image,
%flip it. 
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k_fold = 10;
CVP=cvpartition(n,'kFold',k_fold); %% 10-fold cv

At_cal_hat = zeros(n, p1, p2, p3);
y_hat = zeros(n,1);
err_hat = zeros(n,1);

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

X_cal_train = X_cal(train_flag,:,:,:);
X_cal_test = X_cal(test_flag,:,:,:);

Z_train = Z(train_flag,:);
Z_test = Z(test_flag,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kruskal regression model
X_kruskal_train = tensor(permute(X_cal_train, [2,3,4,1]));

tic;
[beta_hat_test,At_cal_hat_test,glmstats] = kruskal_reg(Z_train,X_kruskal_train,y_train,1,'normal','MaxIter', 10,'Replicates', 1);
toc;

At_cal_hat_test=double(full(At_cal_hat_test)); % p1-p2-p3
At_cal_hat_test = reshape(At_cal_hat_test, [1,p1,p2,p3]);
At_cal_hat_test = repmat(At_cal_hat_test, [n_test,1,1]);

At_cal_hat(test_flag,:,:,:) = At_cal_hat_test;

y_hat_test = sum(X_cal_test .* At_cal_hat_test, [2,3,4]) + Z_test * beta_hat_test;
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



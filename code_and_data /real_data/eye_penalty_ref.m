% eye image model with penalty: use medical eye model to do semi-parametric
% regression.
% implement penalty to identify model structure

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
% do the tensor varying coefficient regression 
A_hat_sm = zeros(n,p_X_mat);
A_unuse_id = [];
h1 = 0.13;
for i=1:n
ker = max(.75*(1-(t-t(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker ~= 0) < 1.4 * (2*p_X_mat+p0)
    A_unuse_id = [A_unuse_id,i];
    continue
end

y_star = y .* ker;
Z_star = Z .* repmat(ker,1,p0);
X_mat_a = X_mat;
sst = (t-t(i))/h1;
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

yte = y(A_use_flag) - sum(X_mat(A_use_flag,:) .* A_hat_sm(A_use_flag, :), 2);
beta_hat_sm = (Z(A_use_flag, :)'*Z(A_use_flag, :) + eye(p0)*(1e-4))\(Z(A_use_flag, :)'*yte);

% s = 5; squeeze(A_hat_sm(:,s,:))'

y_hat_sm = sum(X_mat(A_use_flag,:) .* A_hat_sm(A_use_flag,:), 2) + Z(A_use_flag,:) * beta_hat_sm;
err_hat_sm = y(A_use_flag) - y_hat_sm;

mean(abs(err_hat_sm),'all')
sqrt(mean(err_hat_sm.^2, 'all'))
mean((err_hat_sm- mean(err_hat_sm)).^2, 'all')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B-spline projection
% here we use all time points in spline model
q = 4; n_knot = 6;
knots = linspace(0,1,n_knot);
Bst=bspline_basismatrix(q,[zeros(1,q-1), knots, ones(1,q-1)], t);
L=size(Bst,2);

%spline
Bs = reshape(Bst,[n,1,L]);
X_Bs = X_mat .* Bs;
X_Bs = permute(X_Bs, [3,2,1]);
%tic;
[beta_hat_init,gamma_hat_cp_init,glmstats1] = kruskal_reg(Z,tensor(X_Bs),y,2,'normal');
% toc;
gamma_hat_init=double(full(gamma_hat_cp_init));

A_hat_init = double(ttt(tensor(Bst), tensor(gamma_hat_init),2,1));  % n-by-p_X_mat


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add penalty to distinguish const_0, const_not0, and varying coefficient
% in gamma and beta
tic;
%pentype = 'LASSO';
pentype = 'SCAD';
%pentype = 'MCP';

% set hyper parameters
dif_thrs = 1e-3;
tot_iter = 1000;
ind_thrs = 1e-2;

penparam_beta=0.1;
penparam_gamma_c=0.02;
penparam_gamma_v=0.4;

% step 1: initialize gamma, beta and penalty weight 
X_Bs = double(X_Bs);

gamma_old = gamma_hat_init;
beta_old = beta_hat_init;

dif=1;
iter=0;
while dif>dif_thrs && iter<tot_iter 
iter=iter+1;
% step2: given gamma, update beta
y_tilde = y(A_use_flag) - double(ttt(tensor(X_Bs(:,:,A_use_flag)), tensor(gamma_old), [1,2], [1,2])); % n-by-1
omega_beta = diag(dp(beta_old,penparam_beta,pentype)./(abs(beta_old)+1e-10));
left=Z(A_use_flag,:)'*Z(A_use_flag,:)+n/2*omega_beta;
right=Z(A_use_flag,:)'*y_tilde;
beta_new = left\right;

% step3: given beta, update gamma
% update by part, in each part, update gamma_rs=[gamma_1rs, gamma_2rs, ..,
% gamma_Lrs] L parameters. So here are total RS parts
gamma_new=gamma_old;
for s=1:S
strt_id = (s-1)*R+1;
end_id = s*R;
gamma_s_old=squeeze(gamma_old(:,strt_id:end_id));
y_tilde = y(A_use_flag) - double(ttt(tensor(X_Bs(:,:, A_use_flag)), tensor(gamma_new), [1,2], [1,2])) - Z(A_use_flag,:) * beta_new...
    + double(ttt(tensor(X_Bs(:,strt_id:end_id, A_use_flag)), tensor(gamma_s_old), [1,2],[1,2])); % n-by-1
X_Bs_tilde = double(reshape(X_Bs(:,strt_id:end_id, A_use_flag), [L*R, sum(A_use_flag)]))'; %n-by-LR
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
gamma_new(:,strt_id:end_id)=reshape(gamma_s_new,L,R);
end

dif=sqrt(mean((gamma_new - gamma_old).^2, 'all'));
gamma_old=gamma_new;
beta_old = beta_new;
end

gamma_hat_pen = gamma_old;
beta_hat_pen = beta_old;
A_hat_pen = double(ttt(tensor(Bst), tensor(gamma_hat_pen), 2,1));

toc;

y_hat_pen = sum(X_mat(A_use_flag,:) .* A_hat_pen(A_use_flag,:), 2) + Z(A_use_flag,:) * beta_hat_init;
err_hat_pen = y(A_use_flag) - y_hat_pen;

mean(abs(err_hat_pen),'all')
sqrt(mean(err_hat_pen.^2, 'all'))
mean((err_hat_pen-mean(err_hat_pen)).^2)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find out zero constant coefficient, non-zeros constant coefficient and
% varying coefficients.
flag_beta_nonzero = abs(beta_hat_pen) > ind_thrs;

flag_A_vary = zeros(size(A_hat_pen,2),1);
flag_A_const_nonzero = zeros(size(A_hat_pen,2),1);

for i = 1: size(A_hat_pen,2)
A_hat_pen_i = double(A_hat_pen(A_use_flag,i));
if (sum((A_hat_pen_i - mean(A_hat_pen_i)).^2) >= ind_thrs)
    flag_A_vary(i) = 1;
elseif (abs(mean(A_hat_pen_i)) >= ind_thrs)
    flag_A_const_nonzero(i) = 1;
end
end

flag_A_const_zero = ones(size(A_hat_pen,2),1) - flag_A_const_nonzero - flag_A_vary;


reshape(flag_A_const_zero, R, [])
reshape(flag_A_const_nonzero, R, [])
reshape(flag_A_vary, R, [])
flag_beta_nonzero'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after identify the constant coefficients and varying coefficients
% do a refine regression 
y_ref = y;
Z_ref = Z;
X_ref_mat = X_mat; 

%constant coefficient
beta_nonzero_id = find(flag_beta_nonzero==1);
Z_ref_nonzero = Z_ref(:,beta_nonzero_id);

const_id = find(flag_A_const_nonzero==1);
X_ref_const_nonzero = X_ref_mat(:,const_id);

const_nonzero_ref = [X_ref_const_nonzero, Z_ref_nonzero];

%varying coefficient
vary_id = find(flag_A_vary==1);
X_ref_vary = X_ref_mat(:,vary_id);

A_hat_ref_vary = zeros(n, length(vary_id));
h2=0.13;
for i = 1:n
if any(i == A_unuse_id)
    continue
else
    ker = max(.75*(1-(t-t(i)).^2/(h2^2)),0);
    ker = sqrt(ker/h2);

    y_ref_star = y_ref .* ker;
    const_nonzero_ref_star = const_nonzero_ref .* repmat(ker,1,size(const_nonzero_ref,2));
    % Z_ref_nonzero_star = Z_ref_nonzero .* repmat(ker,1,size(Z_ref_nonzero,2));

    X_ref_a = X_ref_vary;
    sst = (t-t(i))/h2;
    X_ref_b = X_ref_a .* repmat(sst, 1, length(vary_id));
    
    X_ref_a = X_ref_a .* repmat(ker, 1, length(vary_id));
    X_ref_b = X_ref_b .* repmat(ker, 1, length(vary_id));

    XZ_ref_star = [X_ref_a, X_ref_b, const_nonzero_ref_star];
    % XZ_ref_star = [X_ref_a, X_ref_b, Z_ref_nonzero_star];

    % n_row_all0 = sum(all(XZ_star == 0, 2));
    para_hat_ref = (XZ_ref_star' * XZ_ref_star + 1e-4) \ XZ_ref_star' * y_ref_star;
    A_hat_i = para_hat_ref(1:length(vary_id));
    A_hat_ref_vary(i,:) = A_hat_i; 
end
end

yte = y_ref(A_use_flag) - sum(X_ref_vary(A_use_flag,:) .* A_hat_ref_vary(A_use_flag, :), 2);
para_hat_ref_const = (const_nonzero_ref(A_use_flag, :)'*const_nonzero_ref(A_use_flag, :) + eye(size(const_nonzero_ref,2))*(1e-4))\(const_nonzero_ref(A_use_flag, :)'*yte);

beta_hat_ref_nonzero = para_hat_ref_const(end-length(beta_nonzero_id)+1:end);
beta_hat_ref = zeros(p0,1);
beta_hat_ref(beta_nonzero_id) = beta_hat_ref_nonzero;

A_hat_ref_const_nonzero = para_hat_ref_const(1:length(const_id));
A_hat_ref_const_nonzero = repmat(A_hat_ref_const_nonzero',n,1);

A_hat_ref = zeros(n,R*S);
A_hat_ref(:, vary_id) = A_hat_ref_vary;
A_hat_ref(:, const_id) = A_hat_ref_const_nonzero;

y_hat_ref = sum(X_mat(A_use_flag,:) .* A_hat_ref(A_use_flag,:), 2) + Z(A_use_flag,:) * beta_hat_ref;
err_hat_ref = y(A_use_flag) - y_hat_ref;

mean(abs(err_hat_ref),'all')
sqrt(mean(err_hat_ref.^2, 'all'))
mean((err_hat_ref-mean(err_hat_ref)).^2)


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reconstrcut A_cal
A_cal_proj = zeros(n,p1,p2,p3);
for s1 = 1:S1
for s2 = 1:S2
for s3 = 1:S3
s = (s1-1)*S2*S3 + (s2-1)*S3 + s3;
X_cal_s_SV = X_part{s,2};
A_hat_s = squeeze(A_hat_ref(:, (s-1)*R+1:s*R));
A_cal_proj_s = reshape(A_hat_s * X_cal_s_SV, n, p1/S1, p2/S2, p3/S3);
A_cal_proj(:, (s1-1)*(p1/S1)+1:s1*(p1/S1), (s2-1)*(p2/S2)+1:s2*(p2/S2), (s3-1)*(p3/S3)+1:s3*(p3/S3)) = A_cal_proj_s;
end
end
end

% sum(A_cal_proj(124,:,:,:) ~= 0 ,'all')
% imshow(squeeze(A_cal_proj(225,:,:,:)))
%}


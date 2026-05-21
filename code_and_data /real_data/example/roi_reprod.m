clear all;

% load X, Z, t, y
load('roi_reprod.mat')
n = size(X,1);
R = size(X,2);
S = size(X,3);
p0 = size(Z,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do the tensor varying coefficient regression 
X_mat = reshape(X, n, []);

A_hat_sm = zeros(n,R*S);
A_unuse_id = [];
h1 = 0.2;
for i=1:n
ker = max(.75*(1-(t-t(i)).^2/(h1^2)),0);
ker = sqrt(ker/h1);

if sum(ker ~= 0) < 1.2 * (2*R*S+p0)
    A_unuse_id = [A_unuse_id,i];
    continue
end

y_star = y .* ker;
Z_star = Z .* repmat(ker,1,p0);
X_mat_a = X_mat;
sst = (t-t(i))/h1;
X_mat_b = X_mat_a .* repmat(sst, 1, R*S);

X_mat_a = X_mat_a .* repmat(ker, 1, R*S);
X_mat_b = X_mat_b .* repmat(ker, 1, R*S);

XZ_star = [X_mat_a, X_mat_b, Z_star];
% n_row_all0 = sum(all(XZ_star == 0, 2));
para_hat_sm = (XZ_star' * XZ_star + 1e-4) \ XZ_star' * y_star;
A_hat_i = para_hat_sm(1:R*S);
A_hat_sm(i,:) = A_hat_i;
beta_hat_i = para_hat_sm(2*R*S+1:end);
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%constant coefficient
beta_nonzero_id = [5,9,13,23,27,28,32,42,44,49,54];
% beta_nonzero_id = [5,9,13,23,27,28,31,32,33,42,44,48,49,54];
const_id = [18];

%varying coefficient
vary_id = [1,2,3,4,5,6,7,8,11,12,13,15,16,17];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after identify the constant coefficients and varying coefficients
% do a refine regression 
y_ref = y;
Z_ref = Z;
X_ref_mat = X_mat; 

Z_ref_nonzero = Z_ref(:,beta_nonzero_id);
X_ref_const_nonzero = X_ref_mat(:,const_id);
const_nonzero_ref = [X_ref_const_nonzero, Z_ref_nonzero];
X_ref_vary = X_ref_mat(:,vary_id);

A_hat_ref_vary = zeros(n, length(vary_id));
A_unuse_id = [];
h2=0.2;
for i = 1:n
ker = max(.75*(1-(t-t(i)).^2/(h2^2)),0);
ker = sqrt(ker/h2);

if sum(ker ~= 0) < 1.2 * (2*R*S+p0)
    A_unuse_id = [A_unuse_id,i];
    continue
end

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

A_unuse_flag = ismember(1:n, A_unuse_id);
A_use_flag = ~A_unuse_flag;
% sum(A_use_flag)

yte = y_ref(A_use_flag) - sum(X_ref_vary(A_use_flag, :) .* A_hat_ref_vary(A_use_flag, :), 2);
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
This folder contains the code used in the manusctipt "Varying Coefficient Tensor Regression".


-------------------------------------------------------------------------------
The materials are collected in three sub-folders:

(1) toolbox: All code used in our simulation study and real data study is MATLAB code. This folder contains dependent tensor packages we used to conduct our study. 

(2) simulation: files in this folder contains all code used in simulation study. Files starting with "est_" contain codes used to generate experiment results of table 1,2&3 in manuscript. Files starting with "pred_" and "plot_" are corresponding to figure 2&3. We compare our VCTR model with other constant coefficient tensor regression models.

(3) real_data: This folder has all code used to generate our experiment result on real data GRAPE. GRAPE is a public dataset on fundus images and its detailed description can be found in https://doi.org/10.1038/s41597-023-02424-4. eye_select_RS.m is used to select R and S.  "bootstrap" files are used to generate confidence interval and "eye_select_RS.m" file is for model structure identification in CFP and ROI images. Three "pred_" files are used to compare our VCTR model with others on this real data.

-------------------------------------------------------------------------------
Workflow 

Workflow for case 1 & 2 in simulation study

Step 1: 2-D and 3-D data generation. 
Step 2: Do tensor partition and decomposition on each part to extract low-dimensional feature.
Step 3: Use local linear method to estimate functions for tensor covariate and parameters for one-way covariate. (Tuning bandwith)


Workflow for case 3 & 4 in simulation study

Step 1: 2-D and 3-D data generation. 
Step 2: Do tensor partition and decomposition on each part to extract low-dimensional feature.
Step 3: Use local linear method to estimate functions for tensor covariate and parameters for one-way covariate. (Tuning bandwith)
Step 4: Project X and z into B-spline space. (Tuning number of knots and power of spline basics)
Step 5: Include penalty to select variable and identify model structure. (Tuning penalty parameters)
Step 6: Classify different kinds of functions and parameters (Varying, constant non-zero and constant zero).
Step 7: Do a refined regression based on the penalized result, dropping all constant zero function and degenerating constant non-zero functions to parameters.

Workflow for real data

Step 1: Download GRAPE data from https://doi.org/10.6084/m9.figshare.c.6406319.v1
Step 2: Data processing on GRAPE data to extract tensor covariate, one-way covariate and predictors.
Step 3: Select proper number of partitions S and ranks R. 
Step 4: Do tensor partition and decomposition on each part to extract low-dimensional feature.
Step 5: Use local linear method to estimate functions for tensor covariate and parameters for one-way covariate. (Tuning bandwith)
Step 6: Project X and z into B-spline space. (Tuning number of knots and power of spline basics)
Step 7: Include penalty to select variable and identify model structure. (Tuning penalty parameters)
Step 8: Classify different kinds of functions and parameters (Varying, constant non-zero and constant zero).
Step 9: Do a refined regression based on the bootstrap method and the penalized result.

-------------------------------------------------------------------------------
Dependency

Chip: Apple M2
macOS: Sonoma 14.0
Matlab version: MATLAB_R2023a
3 toolboxes are needed: tensor_toolbox, TensorReg, and SparseReg



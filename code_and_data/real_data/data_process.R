library(readxl)
library(ggplot2)
library(dplyr)
library(nlme)
library(cluster)
# library(mice)

# load excel file
baseline_data = read_excel("eye_dataset/VF_and_clinical_information.xlsx",
                           sheet = "processed_baseline")
follow_up_data = read_excel("eye_dataset/VF_and_clinical_information.xlsx",
                            sheet = "processed_follow_up")
baseline_data$Sample = paste(baseline_data$Subject_Number, baseline_data$Laterality, sep = "_")
follow_up_data$Sample = paste(follow_up_data$Subject_Number, follow_up_data$Laterality, sep = "_")
follow_up_data = left_join(follow_up_data, data.frame(Sample=baseline_data$Sample,
                                                      Age=baseline_data$Age,
                                                      Gender=baseline_data$Gender), by = "Sample")

follow_up_data = follow_up_data %>% mutate(Age = Age + Interval_Years)
# delete '/' value in Corresponding_CFP column
follow_up_data = follow_up_data[follow_up_data$Corresponding_CFP != "/", ]

# write.csv(follow_up_data, file = "eye_dataset/follow_up_data.csv", row.names = FALSE)

# generate vector covariate 
vf_cols <- as.character(0:60)
vf_cols <- vf_cols[!vf_cols %in% c("21", "32")]
vct_covar <- data.frame(
  Sample   = follow_up_data$Sample,
  Age      = follow_up_data$Age,
  IsFemale = ifelse(follow_up_data$Gender == "F", 1, 0),
  IOP      = follow_up_data$IOP,
  follow_up_data[, vf_cols],
  CFP      = follow_up_data$Corresponding_CFP,
  check.names = FALSE
)

vf_colnames <- paste0("VF", vf_cols)
colnames(vct_covar)[match(vf_cols, colnames(vct_covar))] <- vf_colnames
vf_cols <- vf_colnames

# drop samples with outlier IOPs
ggplot(vct_covar, aes(x = Age, y = IOP)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "IOP under different samples", x = "Age", y = "IOP")

# boxplot(vct_covar$IOP)
mean_IOP = mean(vct_covar$IOP)
std_IOP = sqrt(var(vct_covar$IOP))
outlier_iop_flag = (vct_covar$IOP > mean_IOP + 2 * std_IOP) | (vct_covar$IOP < mean_IOP - 2 * std_IOP)
drop_sample = unique(vct_covar$Sample[outlier_iop_flag])
vct_covar = vct_covar[!vct_covar$Sample %in% drop_sample, ]

ggplot(vct_covar, aes(x = Age, y = IOP)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),      # Center the title
    panel.grid.major = element_blank(),          # Remove major grid lines
    panel.grid.minor = element_blank(),          # Remove minor grid lines
    panel.border = element_rect(colour = "black", fill = NA), # Add border around the plot
    axis.line = element_blank()                  # Remove axis lines
  ) +
  labs(title = "Intraocular pressure (IOP) with age under different subjects", x = "Age", y = "IOP")

##################################
# impute missing data as mean value 
for (col in vf_cols) {
  mean_val <- mean(vct_covar[[col]][vct_covar[[col]] != -1], na.rm = TRUE)
  vct_covar[[col]][vct_covar[[col]] == -1] <- mean_val
}


# fix outliers with mean value
vf_matrix <- vct_covar[, vf_cols]
vf_mean <- colMeans(vf_matrix, na.rm = TRUE)
vf_sd   <- apply(vf_matrix, 2, sd, na.rm = TRUE)
is_outlier <- abs(sweep(vf_matrix, 2, vf_mean) / vf_sd) > 1.645

for (col in vf_cols) {
  idx <- is_outlier[, col]
  vct_covar[idx, col] <- vf_mean[col]
}


ggplot(vct_covar, aes(x = Age, y = IOP)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA)
  ) +
  labs(title = "Intraocular Pressure (IOP) vs Age", x = "Age", y = "IOP")


ggplot(vct_covar, aes(x = Age, y = IOP)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),      # Center the title
    panel.grid.major = element_blank(),          # Remove major grid lines
    panel.grid.minor = element_blank(),          # Remove minor grid lines
    panel.border = element_rect(colour = "black", fill = NA), # Add border around the plot
    axis.line = element_blank()                  # Remove axis lines
  ) +
  labs(title = "Intraocular pressure (IOP) with age under different subjects", x = "Age", y = "IOP")

ggplot(vct_covar, aes(x = Age)) +
  geom_histogram(binwidth = 5, colour = "black", fill = "white") +  # Adjust binwidth as needed
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),  # Center the title if you have one
    panel.grid.major = element_blank(),      # Remove major grid lines
    panel.grid.minor = element_blank(),      # Remove minor grid lines
    panel.border = element_rect(colour = "black", fill = NA, size = 1), # Add a border
    axis.line = element_line(color = "black") # Ensure axis lines are visible
  ) +
  labs(title = "Histogram of subject's age", x = "Age", y = "Count")


##############################################################################
# draw scatter plot matrix, histogram, and correlation matrix among different VF
library(psych)

par(mgp = c(0.5, 0.4, 0))   
par(cex.axis = 0.7)       
labels <- paste0("VF", 1:length(vf_cols))

vf_data <- vct_covar[, vf_cols]
colnames(vf_data) <- labels

pairs.panels(vf_data[, 51:59],
             method = "pearson",     
             lm = FALSE,            
             density = FALSE,       
             ellipses = FALSE,       
             digits = 2,          
             hist.col = "#00AFBB", 
             pch = 20,             
             cex = 0.05,          
             cex.cor = 1,          
             cex.labels = 1.0,      
             cor = TRUE      
)


##################################
# save file
library(R.matlab)
writeMat(con = "GRAPE_vector.mat", vct_covar = vct_covar)
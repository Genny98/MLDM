# Load required libraries
library(data.table)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3viz)
library(ggplot2)
library(dplyr)

# Set data directory
data.dir <- "/Users/geneviveagyapong/Downloads"

# Function to read datasets
read.dataset <- function(filename, ...) {
  cat("reading", basename(filename), "... ")
  x <- fread(filename, header = TRUE, stringsAsFactors = FALSE, sep = "\t", check.names = FALSE, ...)
  x <- x[match(unique(x[[1]]), x[[1]]), ]
  x <- data.frame(x, row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)
  cat(nrow(x), "x", ncol(x), "\n")
  x
}

# Function to extract participant IDs
extract.participant <- function(id) sub("TCGA-[^-]+-([^-]+)-.*", "\\1", id)

# Load datasets
clinical.dat <- read.dataset(file.path(data.dir, "clinical.txt"))
protein.dat <- read.dataset(file.path(data.dir, "protein.txt"))
mrna.dat <- read.dataset(file.path(data.dir, "mrna.txt"))
mirna.dat <- read.dataset(file.path(data.dir, "mirna.txt"))
methylation.dat <- read.dataset(file.path(data.dir, "methylation.txt"))
mutation.dat <- read.dataset(file.path(data.dir, "mutations.txt"))

# Extract participant IDs and find common IDs
ids.list <- lapply(list(clinical.dat, protein.dat, mrna.dat, mirna.dat, methylation.dat, mutation.dat), function(dat) {
  if (identical(dat, clinical.dat)) {
    return(rownames(dat))
  } else {
    return(extract.participant(colnames(dat)))
  }
})

common.ids <- Reduce(intersect, ids.list)

# Subset datasets based on common IDs
clinical.dat <- clinical.dat[match(common.ids, rownames(clinical.dat)), ]
protein.dat <- protein.dat[, match(common.ids, extract.participant(colnames(protein.dat)))]
mrna.dat <- mrna.dat[, match(common.ids, extract.participant(colnames(mrna.dat)))]
mirna.dat <- mirna.dat[, match(common.ids, extract.participant(colnames(mirna.dat)))]
methylation.dat <- methylation.dat[, match(common.ids, extract.participant(colnames(methylation.dat)))]
mutation.dat <- mutation.dat[, match(common.ids, extract.participant(colnames(mutation.dat)))]

# Preprocess clinical data
clinical.vars <- c("age.at.diagnosis", "estrogen.receptor.status", "progesterone.receptor.status", "lymphocyte.infiltration", "necrosis.percent")
target.var <- "pfi"

clinical.dat <- clinical.dat[, c(target.var, clinical.vars)]
clinical.dat$estrogen.receptor.status <- ifelse(clinical.dat$estrogen.receptor.status == "positive", 1, 0)
clinical.dat$progesterone.receptor.status <- ifelse(clinical.dat$progesterone.receptor.status == "positive", 1, 0)

# Integrate datasets
bc.dat <- data.frame(clinical.dat, t(protein.dat), t(mrna.dat), t(mirna.dat), t(methylation.dat), t(mutation.dat))

# Remove features with > 20% missing values
missing.pct <- sapply(bc.dat, function(v) mean(is.na(v)))
bc.dat <- bc.dat[, missing.pct < 0.2]

# Impute missing values
for (i in which(missing.pct > 0)) {
  missing.idx <- which(is.na(bc.dat[[i]]))
  new.value <- mean(bc.dat[[i]], na.rm = TRUE)
  bc.dat[[i]][missing.idx] <- new.value
}

# Convert 'pfi' to a factor for classification tasks
bc.dat$pfi <- as.factor(bc.dat$pfi)

# Split the data into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(bc.dat)), size = 0.8 * nrow(bc.dat))
train_data <- bc.dat[train_indices, ]
test_data <- bc.dat[-train_indices, ]

# Create tasks
task_train <- TaskClassif$new(id = "breast_cancer_train", backend = data.table(train_data), target = "pfi")
task_test <- TaskClassif$new(id = "breast_cancer_test", backend = data.table(test_data), target = "pfi")

# Preprocess data: Impute missing values and normalize
preprocess_pipeline <- po("imputemedian", param_vals = list(affect_columns = selector_type("numeric"))) %>>%
  po("imputemode", param_vals = list(affect_columns = selector_type("factor"))) %>>%
  po("scale", param_vals = list(center = TRUE, scale = TRUE))

# Define learners with preprocessing pipeline
pipeline_elnet <- preprocess_pipeline %>>% po("learner", lrn("classif.cv_glmnet", predict_type = "prob"))
pipeline_xgb <- preprocess_pipeline %>>% po("learner", lrn("classif.xgboost", predict_type = "prob"))

# Convert to GraphLearner
graph_learner_elnet <- GraphLearner$new(pipeline_elnet)
graph_learner_xgb <- GraphLearner$new(pipeline_xgb)

# Define the resampling strategy
resampling <- rsmp("cv", folds = 5)  # 5-fold cross-validation

# Perform resampling for Elastic Net
tryCatch({
  rr_elnet <- resample(task = task_train, learner = graph_learner_elnet, resampling = resampling)
  print("Elastic Net Resampling Completed")
  print(rr_elnet)
}, error = function(e) {
  print("Error in Elastic Net Resampling")
  print(e)
})

# Perform resampling for XGBoost
tryCatch({
  rr_xgb <- resample(task = task_train, learner = graph_learner_xgb, resampling = resampling)
  print("XGBoost Resampling Completed")
  print(rr_xgb)
}, error = function(e) {
  print("Error in XGBoost Resampling")
  print(e)
})

# Define the measures
measures <- list(msr("classif.auc"), msr("classif.sensitivity"), msr("classif.specificity"))

# Aggregate results for Elastic Net
results_elnet <- rr_elnet$aggregate(measures)
print("Elastic Net Results")
print(results_elnet)

# Aggregate results for XGBoost
results_xgb <- rr_xgb$aggregate(measures)
print("XGBoost Results")
print(results_xgb)

# Combine results for plotting
results <- rbind(
  data.frame(Model = "Elastic Net", Measure = names(results_elnet), Value = as.numeric(results_elnet)),
  data.frame(Model = "XGBoost", Measure = names(results_xgb), Value = as.numeric(results_xgb))
)

# Filter out only the metrics of interest if there are more than needed
results <- results %>% filter(Measure %in% c("classif.auc", "classif.sensitivity", "classif.specificity"))

# Convert measure names to more readable format
results$Measure <- factor(results$Measure, levels = c("classif.auc", "classif.sensitivity", "classif.specificity"), labels = c("AUC", "Sensitivity", "Specificity"))

# Plotting the results
ggplot(results, aes(x = Measure, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  labs(title = "Model Performance Comparison", x = "Performance Metric", y = "Metric Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

# Clinical-only analysis
clinical_only <- clinical.dat[c("pfi", clinical.vars)]  # Selecting only clinical variables
clinical_only$pfi <- as.factor(clinical_only$pfi)  # Ensure the target is appropriate for classification

# Create a Task for clinical data only
clinical_task <- TaskClassif$new(id = "clinical_only", backend = data.table(clinical_only), target = "pfi")

# Elastic Net Pipeline for clinical data only
pipeline_elnet_clinical <- preprocess_pipeline %>>% po("learner", lrn("classif.cv_glmnet", predict_type = "prob"))

# XGBoost Pipeline for clinical data only
pipeline_xgb_clinical <- preprocess_pipeline %>>% po("learner", lrn("classif.xgboost", predict_type = "prob"))

# Convert to GraphLearners for clinical data only
graph_learner_elnet_clinical <- GraphLearner$new(pipeline_elnet_clinical)
graph_learner_xgb_clinical <- GraphLearner$new(pipeline_xgb_clinical)

# Train and evaluate models using cross-validation for clinical data only
rr_elnet_clinical <- resample(clinical_task, graph_learner_elnet_clinical, resampling)
rr_xgb_clinical <- resample(clinical_task, graph_learner_xgb_clinical, resampling)

# Aggregate results for Elastic Net using clinical data only
results_elnet_clinical <- rr_elnet_clinical$aggregate(measures)
print("Elastic Net Clinical Only Results:")
print(results_elnet_clinical)

# Aggregate results for XGBoost using clinical data only
results_xgb_clinical <- rr_xgb_clinical$aggregate(measures)
print("XGBoost Clinical Only Results:")
print(results_xgb_clinical)

# Combine clinical-only results for plotting
results_clinical <- rbind(
  data.frame(Model = "Elastic Net", Measure = names(results_elnet_clinical), Value = as.numeric(results_elnet_clinical)),
  data.frame(Model = "XGBoost", Measure = names(results_xgb_clinical), Value = as.numeric(results_xgb_clinical))
)

# Convert measure names to more readable format
results_clinical$Measure <- factor(results_clinical$Measure, levels = c("classif.auc", "classif.sensitivity", "classif.specificity"), labels = c("AUC", "Sensitivity", "Specificity"))

# Plotting clinical-only results
ggplot(results_clinical, aes(x = Measure, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  labs(title = "Performance Comparison on Clinical Data Only", x = "Performance Metric", y = "Metric Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")



# Compare and conclude
combined_results <- rbind(
  data.frame(Type = "Combined Data", results),
  data.frame(Type = "Clinical Only", results_clinical)
)

# Plot combined results for comparison
ggplot(combined_results, aes(x = Measure, y = Value, fill = Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  facet_wrap(~ Model) +
  labs(title = "Comparison of Model Performance: Combined vs Clinical Only", x = "Performance Metric", y = "Metric Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")


writeLines(capture.output(sessionInfo()), "session_info.txt")

# Install required libraries and load
library(tidyverse)
library(tidymodels)
library(themis)
library(GGally)
library(janitor)
library(discrim)
library(bonsai)
library(klaR)
library(kknn)
library(ranger)
library(xgboost)
library(lightgbm)

# 1. Load the data
koru_data <- read_csv("korucharge_customers.csv") |> clean_names()

# 2. Data Cleaning (Full Dataset for EDA and Final Model)
koru_data_clean_full <- koru_data |>
  mutate(
    weeks_since_signup = parse_number(weeks_since_signup),
    avg_kwh_per_session = parse_number(avg_kwh_per_session),
    peak_share = parse_number(peak_share) / 100,      
    charger_fault_rate = parse_number(charger_fault_rate) / 100,
    avg_wait_minutes = parse_number(avg_wait_minutes),
    weeks_since_last_charge = parse_number(weeks_since_last_charge),
    retained_binary = factor(retained_binary, levels = c("1", "0"), labels = c("Retained", "Churned")),
    across(where(is.logical), as.factor),
    across(where(is.character), as.factor),
    satisfaction_survey = fct_recode(satisfaction_survey, NULL = "NoResponse")
  ) |> 
  dplyr::select(age, gender, region, plan_tier, app_version_lag_weeks, support_ticket, satisfaction_survey,
                notification_opt_in, email_opened_last, app_crashes_30d, weeks_since_signup, home_charging_access,
                charging_sessions_12w, avg_kwh_per_session, peak_share, charger_fault_rate, avg_wait_minutes,
                refund_request_last12w, price_change_exposed, payment_failure_last4w, weeks_since_last_charge,
                user_id, retained_binary)

# 3. Create a Sample for fast model comparison
set.seed(123)
koru_data_clean_sample <- koru_data_clean_full |> slice_sample(n = 10000)







# Churn Rate Overview
churn_summary <- koru_data_clean_full |>
  count(retained_binary) |>
  mutate(prop = n/sum(n))

print(churn_summary)


ibrary(scales)

koru_data_clean_full |> 
  count(retained_binary) |> 
  mutate(prop = n / sum(n)) |> 
  ggplot(aes(x = retained_binary, y = prop, fill = retained_binary)) +
  geom_col(width = 0.6, alpha = 0.9) + 
  geom_text(aes(label = percent(prop, accuracy = 1)), vjust = -0.5, fontface = "bold", size = 5) +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c("Retained" = "#E0E0E0", "Churned" = "#E63946")) +
  labs(
    x = NULL, 
    y = "Customer share (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "none",              
    panel.grid.major.x = element_blank(),  
    axis.text.x = element_text(size = 12, face = "bold") 
  )



###########################################################################
                      # Exploratory Data Analysis
###########################################################################


# Customer Composition by Region (Churned % highlighted)
library(scales)
churn_by_region <- koru_data_clean_full |>
  group_by(region) |> count(retained_binary) |>
  mutate(total_customers = sum(n),
         churn_prop = n/total_customers)

churn_by_region |> 
  ggplot(aes(x = reorder(region, churn_prop * (retained_binary == "Churned")), 
             y = churn_prop, 
             fill = retained_binary)) +
  geom_col(width = 0.75, alpha = 0.9) + 
  coord_flip() +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = c("Retained" = "#E0E0E0", "Churned" = "#E63946")) +
  geom_hline(yintercept = mean(koru_data_clean_full$retained_binary == "Churned"), 
             linetype = "dashed", color = "#8B0000", linewidth = 0.8) +
  labs(
    x = NULL, 
    y = "Customer share within region (%)", 
    fill = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 11)
  )


# Churn Rate (Churned %) by Plan Tier

churn_by_plan <- koru_data_clean_full |>
  group_by(plan_tier) |> count(retained_binary) |>
  mutate(total_customers = sum(n),
         churn_prop = n/total_customers)

churn_by_plan |> 
  ggplot(aes(x = reorder(plan_tier, -churn_prop * (retained_binary == "Churned")), 
             y = churn_prop, 
             fill = retained_binary)) +
  geom_col(width = 0.6, alpha = 0.9) +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = c("Retained" = "#E0E0E0", "Churned" = "#E63946")) +
  labs(
    x = NULL, 
    y = "Proportion of Customers",
    fill = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(size = 12, face = "bold")
  )


# Average Wait Time (minutes) for Churned vs Retained Customers

ggplot(koru_data_clean_full, aes(x = retained_binary, y = avg_wait_minutes, fill = retained_binary)) +
  geom_boxplot(alpha = 0.9, width = 0.5) +
  scale_fill_manual(values = c("Retained" = "#E0E0E0", "Churned" = "#E63946")) +
  labs(
    x = NULL,
    y = "Average Wait (Minutes)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 12, face = "bold")
  )


# Churn Rate by Price Change Exposure (%)

koru_data_clean_full |>
  ggplot(aes(x = price_change_exposed, fill = retained_binary)) +
  geom_bar(position = "fill", width = 0.6, alpha = 0.9) +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = c("Retained" = "#E0E0E0", "Churned" = "#E63946")) +
  labs(
    y = "Customer share within group (%)", 
    x = "Exposed to Price Change",
    fill = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(size = 12, face = "bold")
  )



###########################################################################
                            # Modelling
###########################################################################

# 1. Split the data (70/30 as per sample)
set.seed(010)
churn_split <- initial_split(koru_data_clean_sample, prop = 0.70, strata = retained_binary)
train_data <- training(churn_split)
test_data  <- testing(churn_split)

# 2. Create Cross-Validation Folds
cv_folds <- vfold_cv(train_data, v = 5, strata = retained_binary)

# 3. The Recipe
churn_recipe <- recipe(retained_binary ~ ., data = train_data) |>
  update_role(user_id, new_role = "ID") |>
  # Handle missing values in survey if any (step_impute_mode or median)
  step_impute_mode(satisfaction_survey) |>
  # Interaction term requested by Case Study: waits x faults x peak usage
  step_interact(terms = ~ avg_wait_minutes:charger_fault_rate:peak_share) |>
  # Pre-processing steps from sample
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  # Upsample because Churn is usually the minority
  step_upsample(retained_binary, over_ratio = 1)

########### Define Model Specifications ###########

# 1. Logistic Regression
lr_model <- logistic_reg() |> set_engine("glm")

# 2. Naive Bayes
nb_model <- naive_Bayes() |> set_engine("klaR")

# 3. K-Nearest Neighbours
knn_model <- nearest_neighbor(neighbors = 5) |> 
  set_engine("kknn") |> 
  set_mode("classification")

# 4. Random Forest
rf_model <- rand_forest(trees = 1000) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("classification")

# 5. XGBoost
xgb_model <- boost_tree() |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# 6. LightGBM
lgbm_model <- boost_tree() |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

########### Create Workflows ###########

lr_wflow   <- workflow() |> add_model(lr_model)   |> add_recipe(churn_recipe)
nb_wflow   <- workflow() |> add_model(nb_model)   |> add_recipe(churn_recipe)
knn_wflow  <- workflow() |> add_model(knn_model)  |> add_recipe(churn_recipe)
rf_wflow   <- workflow() |> add_model(rf_model)   |> add_recipe(churn_recipe)
xgb_wflow  <- workflow() |> add_model(xgb_model)  |> add_recipe(churn_recipe)
lgbm_wflow <- workflow() |> add_model(lgbm_model) |> add_recipe(churn_recipe)

########### Define the Metric Set ###########

churn_metrics <- metric_set(
  accuracy,
  roc_auc,
  metric_tweak("sens2", sens, event_level = "second"),
  metric_tweak("spec2", spec, event_level = "second")
)

########### Fit Models to Training Data (Cross-Validation) ###########

# Control object to save predictions for ROC curves later
ctrl <- control_grid(save_pred = TRUE, parallel_over = "everything")

# Run resamples for each
lr_res   <- lr_wflow   |> fit_resamples(resamples = cv_folds, metrics = churn_metrics, control = ctrl)
nb_res   <- nb_wflow   |> fit_resamples(resamples = cv_folds, metrics = churn_metrics, control = ctrl)
knn_res  <- knn_wflow  |> fit_resamples(resamples = cv_folds, metrics = churn_metrics, control = ctrl)
rf_res   <- rf_wflow   |> fit_resamples(resamples = cv_folds, metrics = churn_metrics, control = ctrl)
xgb_res  <- xgb_wflow  |> fit_resamples(resamples = cv_folds, metrics = churn_metrics, control = ctrl)
lgbm_res <- lgbm_wflow |> fit_resamples(resamples = cv_folds, metrics = churn_metrics, control = ctrl)

# Combine all results into one table
all_results <- bind_rows(
  lr_res   |> collect_metrics() |> mutate(model = "Logistic Regression"),
  nb_res   |> collect_metrics() |> mutate(model = "Naive Bayes"),
  knn_res  |> collect_metrics() |> mutate(model = "KNN"),
  rf_res   |> collect_metrics() |> mutate(model = "Random Forest"),
  xgb_res  |> collect_metrics() |> mutate(model = "XGBoost"),
  lgbm_res |> collect_metrics() |> mutate(model = "LightGBM")
)

all_pred <-
  bind_rows(
    lr_res |> collect_predictions() |> mutate(model = "Logistic Regression"),
    nb_res |> collect_predictions() |> mutate(model = "Naive Bayes"),
    knn_res |> collect_predictions() |> mutate(model = "KNN"),
    rf_res |> collect_predictions() |> mutate(model = "Random Forest"),
    xgb_res |> collect_predictions() |> mutate(model = "XGBoost"),
    lgbm_res |> collect_predictions() |> mutate(model = "LightGBM")
  )


########### Cross-Validation Performance Metrics by Algorithm ###########

library(knitr)
library(tidyr)
library(dplyr)

perf_table <- all_results |>
  as.data.frame() |> 
  filter(.metric %in% c("accuracy", "roc_auc", "sens2", "spec2")) |>
  group_by(model, .metric) |>
  summarise(mean = mean(mean, na.rm = TRUE), .groups = "drop") |>
  tidyr::pivot_wider(names_from = .metric, values_from = mean) |>
  rename(
    Model = model,
    `Accuracy` = accuracy,
    `ROC AUC` = roc_auc,
    `Sensitivity (Churn)` = sens2,
    `Specificity (Retain)` = spec2
  ) |>
  arrange(desc(`ROC AUC`))

kable(perf_table, format = "markdown", align = "lcccc", digits = 3)



########### ROC Curves by Fold for Evaluated Algorithms ###########

all_pred |>
  group_by(id,model) |>   # id contains the folds
  roc_curve(retained_binary, .pred_Churned, event_level = "second") |>
  autoplot(aes(col = model)) + 
  facet_wrap(facets = vars(model)) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y = element_text(size = 8) 
  )


########### Final Model Fit ###########

# Split the data (70/30 on original data)
set.seed(032)
churn_split_final <- initial_split(koru_data_clean_full, prop = 0.70, strata = retained_binary)
train_data_full <- training(churn_split_final)
test_data_full  <- testing(churn_split_final)

# Create Cross-Validation Folds
cv_folds_full <- vfold_cv(train_data_full, v = 5, strata = retained_binary)

# The Recipe
churn_recipe_final <- recipe(retained_binary ~ ., data = train_data_full) |>
  update_role(user_id, new_role = "ID") |>
  step_impute_mode(satisfaction_survey) |>
  step_interact(terms = ~ avg_wait_minutes:charger_fault_rate:peak_share) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_upsample(retained_binary, over_ratio = 1)

# Create New LR Workflow
lr_wflow_final <- workflow() |>
  add_model(lr_model)  |>
  add_recipe(churn_recipe_final)

# Finalize the workflow with LR model
final_lr_fit <- last_fit(lr_wflow_final, churn_split_final, metrics = churn_metrics)

final_fitted_model <- extract_workflow(final_lr_fit)

lr_pred_final <- predict(final_fitted_model, test_data_full, type = "prob") |>
  bind_cols(
    predict(final_fitted_model, test_data_full, type = "class"),
    test_data_full |> dplyr::select(retained_binary) 
  )


########### ROC Curve for Final Logistic Regression Model ###########

# ROC curve
lr_pred_final |>
  roc_curve(truth = retained_binary,
            .pred_Churned,
            event_level = "second") |>
  autoplot()


########### Confusion Matrix on the Final Test Dataset ###########

final_lr_fit |>
  collect_predictions() |>
  conf_mat(truth = retained_binary, estimate = .pred_class) |>
  autoplot(type = "heatmap") 


###########################################################################
                          # Final Findings
        # Variable Importance: Top Drivers of Churn vs. Retention
###########################################################################

top_vars <- final_lr_fit |>
  extract_fit_parsnip() |>
  tidy() |>
  filter(term != "(Intercept)") |>
  arrange(p.value) 

# 1. Prepare the data for the Top 10 only (5 churn, 5 retention)
top_10_drivers <- top_vars |>
  mutate(
    odds_ratio = exp(estimate),
    direction = ifelse(estimate > 0, "Increase churn", "Reduce churn"),
    # Calculate absolute impact to find the strongest drivers regardless of direction
    abs_impact = abs(estimate) 
  ) |>
  group_by(direction) |>
  # Take the 5 strongest for each group
  slice_max(abs_impact, n = 5) |> 
  ungroup()

# 2. Create the plot
ggplot(top_10_drivers,
       aes(x = reorder(term, odds_ratio),
           y = odds_ratio,
           fill = direction)) +
  geom_col(width = 0.7, alpha = 0.9) +
  coord_flip() +
  scale_y_log10() +
  scale_fill_manual(values = c("Increase churn" = "#e41a1c", "Reduce churn" = "#377eb8")) +
  labs(
    subtitle = "Odds Ratio > 1 increases churn risk | Odds Ratio < 1 promotes retention",
    x = "Variable",
    y = "Odds Ratio (Impact Strength on Log Scale)"
  ) +
  theme_minimal() +
  theme(
    plot.subtitle = element_text(hjust = 0, size = 10),
    plot.title.position = "plot",  
    legend.title = element_blank(),
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 10)
  )
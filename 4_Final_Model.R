# Final Model #

cat("\014")
rm(list = ls())

library(tidyverse)
library(caret)
library(xgboost)
library(forcats)
library(lubridate)
library(ggplot2)
library(knitr)

df <- read.csv("Cleaned_London.csv", stringsAsFactors = TRUE)

# 1. Enhancements: Log Price, Reduced Neighbourhood, Demand Feature
# 1.1 Repeat Data Cleaning
df <- df %>% drop_na()

# 1.2 Log-transforming the target variable
df$log_price <- log1p(df$price)

# 1.3 Reducing cardinality of neighbourhood feature
top10_nhoods <- names(sort(table(df$neighbourhood_cleansed), decreasing = TRUE))[1:10]
df$neighbourhood_top <- ifelse(df$neighbourhood_cleansed %in% top10_nhoods,
                               as.character(df$neighbourhood_cleansed), "Other") %>% as.factor()


# 2. Training and Testing the new XGBoost model
# 2.1 Preparing Data for XGBoost
factor_vars <- df %>% select(where(is.factor)) %>% names()
df <- df %>% mutate(across(all_of(factor_vars), as.character)) %>%
  mutate(across(all_of(factor_vars), as.factor))

df_dummies <- model.matrix(log_price ~ . - price -1, data = df)
X <- scale(df_dummies)
y <- df$log_price

# 2.2 Train-Test Split
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test  <- X[-train_index, ]
y_test  <- y[-train_index]

# 2.3 Traiing XGBoost Model (Balanced Defaults)
set.seed(42)
xgb_model <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 200,
  max_depth = 6,
  eta = 0.1,
  objective = "reg:squarederror",
  verbose = 0
)

# 2.4 Test Prediction
y_pred_log <- predict(xgb_model, as.matrix(X_test))

# 3. Evaluating the Model
# 3.1 Metrics (log scale)
mse_log <- mean((y_test - y_pred_log)^2)
rmse_log <- sqrt(mse_log)

# 3.2 Converting back to actual prices
y_test_real <- expm1(y_test)
y_pred_real <- expm1(y_pred_log)

# 3.3 Final Evaluation Metrics 
mse_real <- mean((y_test_real - y_pred_real)^2)
rmse_real <- sqrt(mse_real)
r2_real <- cor(y_test_real, y_pred_real)^2

cat("MSE (actual price):", round(mse_real, 2), "\n")
cat("RMSE (actual price):", round(rmse_real, 2), "\n")
cat("R² (actual price):", round(r2_real, 4), "\n")


# 6. Visualising Results
# 6.1 Predicted vs Actual
Figure10 <- ggplot(data = data.frame(Actual = y_test_real, Predicted = y_pred_real),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Figure 10: Predicted vs Actual Prices (Enhanced XGBoost)",
    x = "Actual Price (£)", y = "Predicted Price (£)"
  ) +
  theme_minimal()

print(Figure10)

# 6.2 Feature Importance
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix[1:10], main = "Figure 11: Top 10 Most Important Features (Enhanced XGBoost)")
Figure11 <- recordPlot()

replayPlot(Figure11)

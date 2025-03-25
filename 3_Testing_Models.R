# Testing Models #

cat("\014")   
rm(list = ls())  

# Airbnb Price Prediction: Regression Trees, Random Forest, XGBoost
library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(rpart)
library(rpart.plot)
library(forcats)
library(lubridate)
library(ggplot2)
library(knitr)
library(kableExtra)

df <- read.csv("Cleaned_London.csv", stringsAsFactors = TRUE)

## 1. Additional Cleaning:
# 1.1 Selecting relevant features (excluding non-predictive ones)
df <- df %>% drop_na()

# 1.2 Reducing High-Cardinality Categorical Variables
df$neighbourhood_cleansed <- fct_lump(df$neighbourhood_cleansed, n = 7)  # Keeps top 7, merge others into "Other"


## 2. Validation Set Approach: Regression Trees
# 2.1 Creating the Train-Test Split (80/20)
set.seed(42)
train_index <- createDataPartition(df$price, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# 2.2 Fitting and plotting Regression Tree Model
set.seed(42)
rpart_tree <- rpart(price ~ ., data = train_data, method = "anova")
rpart.plot(rpart_tree, main = "Figure 5: Pruned Regression Tree for Price Prediction")

Figure5 <- recordPlot()

replayPlot(Figure5)

# 2.3 Predictions + Evaluation
train_preds_tree <- predict(rpart_tree, newdata = train_data)
test_preds_tree <- predict(rpart_tree, newdata = test_data)

train_mse_tree <- mean((train_preds_tree - train_data$price)^2)
test_mse_tree  <- mean((test_preds_tree - test_data$price)^2)

cat("Training MSE Tree:", round(train_mse_tree, 2), "\n")
cat("Test MSE Tree:", round(test_mse_tree, 2), "\n")

# 3. Ensemble Methods: Random Forest and XGBoost
# 3.1 Preparation: Converting factors to dummy variables
factor_vars <- df %>% select(where(is.factor)) %>% names()
df <- df %>% mutate(across(all_of(factor_vars), as.character)) %>%
  mutate(across(all_of(factor_vars), as.factor))

df_dummies <- model.matrix(price ~ . -1, data = df)  # Remove intercept
X <- scale(df_dummies)
y <- df$price

# 3.2 Splititng into Train/Test Sets Again for Ensemble Models
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# 3.3 Training Models
set.seed(42)
rf_model <- randomForest(X_train, y_train, ntree = 100)
gbr_model <- xgboost(data = as.matrix(X_train), label = y_train, nrounds = 100, objective = 'reg:squarederror')
lasso_model <- cv.glmnet(as.matrix(X_train), y_train, alpha = 1) # LASSO used to compare to linear methods

# 3.4 Predictions
y_pred_rf <- predict(rf_model, X_test)
y_pred_gbr <- predict(gbr_model, as.matrix(X_test))
y_pred_lasso <- predict(lasso_model, as.matrix(X_test), s = "lambda.min")

# 3.5 Evaluation 
mse <- function(actual, predicted) mean((actual - predicted)^2)
r2 <- function(actual, predicted) cor(actual, predicted)^2

results <- data.frame(
  Model = c("Regression Tree", "Random Forest", "Gradient Boosting", "LASSO"),
  MSE = c(test_mse_tree, mse(y_test, y_pred_rf), mse(y_test, y_pred_gbr), mse(y_test, y_pred_lasso)),
  RMSE = c(sqrt(test_mse_tree), sqrt(mse(y_test, y_pred_rf)), sqrt(mse(y_test, y_pred_gbr)), sqrt(mse(y_test, y_pred_lasso))),
  R2 = c(r2(test_data$price, test_preds_tree), r2(y_test, y_pred_rf), r2(y_test, y_pred_gbr), r2(y_test, y_pred_lasso))
)

# 3.6 Table of Results
kable(results, caption = "Model Performance Comparison")
results %>%
  kbl(caption = "Figure 6: Model Performance Comparison") %>%
  kable_styling(latex_options = c("striped", "hold_position", "booktabs"))
Figure6 <- recordPlot()

replayPlot(Figure6)

# 4. Evaluating XGBoost Model
# 4.1 Interpreting MSE and RMSE
xgb_mse <- mean((y_test - y_pred_gbr)^2)
xgb_rmse <- sqrt(xgb_mse)

cat("XGBoost MSE:", round(xgb_mse, 2), "\n")
cat("XGBoost RMSE:", round(xgb_rmse, 2), "\n")

# 4.2 Visualizing Actual Price Data
Figure7 <- ggplot(df, aes(x = price)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "white") +
  labs(
    title = "Figure 7: Distribution of Airbnb Listing Prices in London",
    x = "Price (£)", y = "Number of Listings"
  ) +
  theme_minimal()

print(Figure7)

# 4.3 Predicted vs Actual Prices
Figure8 <- ggplot(data = data.frame(Actual = y_test, Predicted = y_pred_gbr), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Figure 8: Predicted vs Actual Prices (XGBoost)",
    x = "Actual Price (£)",
    y = "Predicted Price (£)"
  ) +
  theme_minimal()

print(Figure8)

# 4.4 Top 10 Most Important Features
importance_matrix <- xgb.importance(model = gbr_model)
xgb.plot.importance(importance_matrix[1:10], main = "Figure 9: Top 10 Most Important Features (XGBoost)")
Figure9 <- recordPlot()
replayPlot(Figure9)

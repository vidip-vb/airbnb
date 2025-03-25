# Data Exploration #

cat("\014")   
rm(list = ls())  

library(tidyverse)
library(car)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(lmtest)

df <- read.csv("Cleaned_London.csv", stringsAsFactors = TRUE)

# Data Exploration: Linear Modelling

## 1. Identifying Key Numeric Features (Pearson Correlation)**
numeric_df <- df %>% select_if(is.numeric)

# 1.1 Computing Pearson correlation with price
cor_matrix <- cor(numeric_df, use = "complete.obs")

# 1.2 Selecting top 5 most correlated numeric features with price
top_numeric_features <- names(sort(abs(cor_matrix["price", ]), decreasing = TRUE))[2:6]  
print(top_numeric_features)

## 2. Identifying Key Categorical Features (Spearman Correlation)**
# 2.1 Selecting categorical/ordinal variables (including factor and ordered factor)
categorical_df <- df %>% select(where(is.factor))

# 2.2 Converting factors to numeric ranks
ranked_df <- categorical_df %>% mutate_all(~as.numeric(as.factor(.)))

# 2.3 Adding price column back to ranked dataset
ranked_df$price <- df$price   

# 2.4 Computing Spearman correlation matrix using ranked categorical values
spearman_matrix <- cor(ranked_df, method = "spearman", use = "pairwise.complete.obs")

# 2.5 Selecting top 5 categorical features most correlated with price
top_categorical_features <- names(sort(abs(spearman_matrix["price", ]), decreasing = TRUE))[2:6]  
print(top_categorical_features)

## 3. Fitting a basic linear model with selected features**
# 3.1 Constructing formula dynamically
formula <- as.formula(paste("price ~", paste(c(top_numeric_features, top_categorical_features), collapse = " + ")))

# 3.2 Fitting linear model with selected features
linear_model <- lm(formula, data = df)
summary(linear_model)

## 4. Residuals vs. Fitted Plot (LOESS Curve)
Figure1 <- ggplot(data = data.frame(Fitted = fitted(linear_model), Residuals = resid(linear_model)), 
       aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "blue") + 
  geom_smooth(method = "loess", color = "red", se = FALSE) + 
  labs(title = "Figure 1: Residuals vs Fitted Values (Linear Model)",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

print(Figure1)

# - The LOESS Curve shows potential nonlinearity in the dataset shown buy the curved (wave-like) structure. A linear model assumes that residuals are randomly scattered around zero, so the LOESS curve indicates otherwise.

## 5. Ramsey RESET Test (Formal test for Linearity)
reset_test <- resettest(linear_model, power = 2, type = "fitted")
print(reset_test)

# - Finding potential nonlinearity in the dataset means that we will be better off using regression trees. Step 6 and on looks at whether RTs do better when handling the London Airbnb dataset.

## 6. Exploratory Regression Tree to Capture Nonlinearity

library(rpart)
library(rpart.plot)
library(caret)

# 6.1 Removing rows with missing target
df_model <- df %>% filter(!is.na(price))

# 6.2 Imputing missing values
# - For Numeric features, median values are used
numeric_vars <- df_model %>% select(where(is.numeric)) %>% names()
df_model[numeric_vars] <- lapply(df_model[numeric_vars], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# - For Categorical features, the mode is used
factor_vars <- df_model %>% select(where(is.factor)) %>% names()
for (f in factor_vars) {
  mode_val <- names(sort(table(df_model[[f]]), decreasing = TRUE))[1]
  df_model[[f]][is.na(df_model[[f]])] <- mode_val
}

# 6.3 One-hot encoding for categorical features
dummies <- dummyVars(price ~ ., data = df_model, fullRank = TRUE)
df_encoded <- predict(dummies, newdata = df_model) %>% as.data.frame()
df_encoded$price <- df_model$price

# 6.4 Fitting a readable regression tree (no pruning yet)
rt_model <- rpart(price ~ ., data = df_encoded, method = "anova",
                  control = rpart.control(cp = 0.005, minsplit = 100, maxdepth = 6))

# 6.5 Plotting the tree
rpart.plot(rt_model, type = 2, extra = 101, tweak = 1.2,
           main = "Figure 2: Regression Tree for Airbnb Prices")


Figure2 <- recordPlot()

replayPlot(Figure2)

rtpreds <- predict(rt_model, newdata = df_encoded)
# 6.6 Residuals vs Fitted Values (Regression Tree)
residuals_rt <- df_encoded$price - rtpreds

Figure3 <- ggplot(data = data.frame(Fitted = rtpreds, Residuals = residuals_rt),
       aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  labs(
    title = "Figure 3: Residuals vs Fitted Values (Regression Tree)",
    x = "Fitted Values",
    y = "Residuals"
  ) +
  theme_minimal()

print(Figure3)

# 6.7 Plotting predicted vs actual prices (visual fit)

Figure4 <- ggplot(data = data.frame(Actual = df_encoded$price, Predicted = rtpreds),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Figure 4: Predicted vs Actual Prices (Regression Tree)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal()

print(Figure4)
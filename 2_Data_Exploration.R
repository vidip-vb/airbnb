rm(list = ls())

# Load necessary libraries
library(tidyverse)
library(car)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(lmtest)

df <- read.csv("Cleaned_London.csv", stringsAsFactors = TRUE)

# Data Exploration: Linear Modelling

### Step 1: Identifying Key Numeric Features (Pearson Correlation)**
numeric_df <- df %>% select_if(is.numeric)

# 1.1 Computing Pearson correlation with price
cor_matrix <- cor(numeric_df, use = "complete.obs")

# 1.2 Selecting top 5 most correlated numeric features with price
top_numeric_features <- names(sort(abs(cor_matrix["price", ]), decreasing = TRUE))[2:6]  
print(top_numeric_features)

### Step 2: Identifying Key Categorical Features (Spearman Correlation)**
# Selecting categorical/ordinal variables (including factor and ordered factor)
categorical_df <- df %>% select(where(is.factor))

# 2.1 Converting factors to numeric ranks
ranked_df <- categorical_df %>% mutate_all(~as.numeric(as.factor(.)))

# 2.2 Adding price column back to ranked dataset
ranked_df$price <- df$price   

# 2.3 Computing Spearman correlation matrix using ranked categorical values
spearman_matrix <- cor(ranked_df, method = "spearman", use = "pairwise.complete.obs")

# 2.4 Selecting top 5 categorical features most correlated with price
top_categorical_features <- names(sort(abs(spearman_matrix["price", ]), decreasing = TRUE))[2:6]  
print(top_categorical_features)

### Step 3: Fitting a basic linear model with selected features**
# Constructing formula dynamically
formula <- as.formula(paste("price ~", paste(c(top_numeric_features, top_categorical_features), collapse = " + ")))

# 3.` Fitting linear model with selected features
linear_model <- lm(formula, data = df)

# 3.2 Display model summary
summary(linear_model)

### Step 4: Residuals vs. Fitted Plot (LOESS Curve)
ggplot(data = data.frame(Fitted = fitted(linear_model), Residuals = resid(linear_model)), 
       aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "blue") + 
  geom_smooth(method = "loess", color = "red", se = FALSE) +  # Loess curve for pattern detection
  labs(title = "Residuals vs Fitted Values (Checking for Nonlinearity)",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

# - The plot shows potential nonlinearity in the dataset shown buy the curved (wave-like) structure. A linear model assumes that residuals are randomly scattered around zero, so the LOESS curve indicates otherwise.

### Step 5: Ramsey RESET Test (Formal test for Linearity)
reset_test <- resettest(linear_model, power = 2, type = "fitted")
print(reset_test)

# - Finding potential nonlinearity in the dataset means that we will be better off using decision trees. Step 6 and on looks at whether DTs do better when handling the London Airbnb dataset.

### Step 6: Exploratory Decision Tree to Capture Nonlinearity

library(rpart)
library(rpart.plot)
library(caret)

# 6.1: Removing irrelevant columns
exclude_cols <- c("name", "host_since", "first_review", "last_review", "latitude", "longitude")
df_model <- df %>% select(-all_of(exclude_cols))

# 6.2: Removing rows with missing target
df_model <- df_model %>% filter(!is.na(price))

# 6.3: Imputing missing values
# Numeric: median
numeric_vars <- df_model %>% select(where(is.numeric)) %>% names()
df_model[numeric_vars] <- lapply(df_model[numeric_vars], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Categorical: mode
factor_vars <- df_model %>% select(where(is.factor)) %>% names()
for (f in factor_vars) {
  mode_val <- names(sort(table(df_model[[f]]), decreasing = TRUE))[1]
  df_model[[f]][is.na(df_model[[f]])] <- mode_val
}

# 6.4: One-hot encoding for categorical features
dummies <- dummyVars(price ~ ., data = df_model, fullRank = TRUE)
df_encoded <- predict(dummies, newdata = df_model) %>% as.data.frame()
df_encoded$price <- df_model$price

# 6.5: Fitting a readable decision tree (no pruning yet)
dt_model <- rpart(price ~ ., data = df_encoded, method = "anova",
                  control = rpart.control(cp = 0.005, minsplit = 100, maxdepth = 6))

# 6.6: Plotting the tree
rpart.plot(dt_model, type = 2, extra = 101, tweak = 1.2,
           main = "Exploratory Decision Tree for Airbnb Prices")

# 6.7: Plotting predicted vs actual prices (visual fit)
dtpreds <- predict(dt_model, newdata = df_encoded)

ggplot(data = data.frame(Actual = df_encoded$price, Predicted = dtpreds),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Decision Tree: Predicted vs Actual Prices",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal()


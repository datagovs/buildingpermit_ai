library(data.table)
library(rsample)  # data splitting 
library(glmnet)   # implementing regularized regression approaches
library(dplyr)    # basic data manipulation procedures
library(ggplot2)  # plotting
library(broom)

bldg_permit_filename <- file.path(
  Sys.getenv('MIAMIVDD_DATA_PATH'),
  'Building_Permits_Issued_By_City_Of_Miami_From_2014_To_Present.csv'
)

bldg_permit_dt <- fread(bldg_permit_filename, sep=",")
bldg_permit_dt$ScopeofWork <- as.factor(bldg_permit_dt$ScopeofWork)

bldg_permit_fit_dt <- bldg_permit_dt[
  !is.na(bldg_permit_dt$TotalDaysInPlanReviewNumeric)
  & !is.na(bldg_permit_dt$ScopeofWork)
  & !is.na(bldg_permit_dt$TotalSQFT)
  ]

fit_dt <- bldg_permit_fit_dt[,list(TotalDaysInPlanReviewNumeric)]
fit_dt <- cbind(fit_dt, one_hot(bldg_permit_fit_dt[, c('ScopeofWork')]))

split <- initial_split(fit_dt, prop = .7)
summary(fit_dt)

train <- training(split)
test  <- testing(split)

train_x <- model.matrix(TotalDaysInPlanReviewNumeric ~ ., train)[, -1]
train_y <- train$TotalDaysInPlanReviewNumeric

test_x <- model.matrix(TotalDaysInPlanReviewNumeric ~ ., test)[, -1]
test_y <- test$TotalDaysInPlanReviewNumeric

dim(train_x)
length(train_y)
dim(test_x)
length(test_y)

model <- cv.glmnet(
  x = train_x,
  y = train_y,
  alpha = 0
)

model

min(model$cvm)       # minimum MSE

plot(model, xvar = "lambda")

# feature importance
coef(model, s = "lambda.1se") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  top_n(25, wt = abs(value)) %>%
  ggplot(aes(value, reorder(row, value))) +
  geom_point() +
  ggtitle("Top 25 influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

# r squared
rsq = 1 - model$cvm/var(train_y)
plot(model$lambda, rsq)

# predicted vs actual
pred <- predict(model, s = model$lambda.min, test_x)
plot(test_y, pred)
mean((test_y - pred)^2)

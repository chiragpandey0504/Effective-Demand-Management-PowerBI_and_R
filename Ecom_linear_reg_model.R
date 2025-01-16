## Importing libraries:

library(plotly)
library(dplyr)
library(tidyverse)
library(olsrr)
library(caTools)
library(xgboost)
library(data.table)
library(ggplot2)
library(caret)

## Loading data:
Looker_Ecommerce=read.csv("Data.csv")

nrow(Looker_Ecommerce)
ncol(Looker_Ecommerce)

summary(Looker_Ecommerce)

colnames(Looker_Ecommerce)

str(Looker_Ecommerce)

## Variable selection:
Looker_Ecommerce = subset(Looker_Ecommerce,select = -c(1,2,3,4,5,6,7,8,9,11,12,16,17,19,20,21,22,23,27,30,31,32,33,34,35,36,37,38))
colnames(Looker_Ecommerce)

## Checking NA values:
colSums(is.na(Looker_Ecommerce))

## Outlier removal:
boxplot(Looker_Ecommerce$product.sale_price)
range(Looker_Ecommerce$product.sale_price)
Q1 <- quantile(Looker_Ecommerce$product.sale_price, .25)
Q3 <- quantile(Looker_Ecommerce$product.sale_price, .75)
IQR <- IQR(Looker_Ecommerce$product.sale_price)

Looker_Ecommerce <- subset(Looker_Ecommerce, Looker_Ecommerce$product.sale_price> 
                             (Q1 - 1.5*IQR) & Looker_Ecommerce$product.sale_price< 
                             (Q3 + 1.5*IQR))
range(Looker_Ecommerce$product.sale_price)
str(Looker_Ecommerce)


## Converting categorical variables in Factor variable:
Looker_Ecommerce$users.gender = as.factor(Looker_Ecommerce$users.gender)
Looker_Ecommerce$users.state = as.factor(Looker_Ecommerce$users.state)
Looker_Ecommerce$users.country = as.factor(Looker_Ecommerce$users.country)
Looker_Ecommerce$category = as.factor(Looker_Ecommerce$category)
Looker_Ecommerce$products.brand = as.factor(Looker_Ecommerce$products.brand)



## Splitting Data into Train and Test:
set.seed(1)
sample <- sample.split(Looker_Ecommerce$product.sale_price, SplitRatio = 0.7)
train  <- subset(Looker_Ecommerce, sample == TRUE)
test   <- subset(Looker_Ecommerce, sample == FALSE)
dim(train)
dim(test)


## Linear regression model:
model1 <- lm(product.sale_price~., data = train)

## Calculate R-squared on training data:
summary(model1)

test$products.brand[which(!(test$products.brand %in% unique(train$products.brand)))] <- NA
test = na.omit(test)

## Calculate R-squared on testing data:
Predictions = predict(model1,newdata = test)
error=test$product.sale_price - Predictions
errorSq=error^2
SSE=sum(errorSq)
mue=mean(train$product.sale_price)
error2=test$product.sale_price - mue
error2Sq=error2^2
SST=sum(error2Sq)
R2=1-SSE/SST
R2

##################################################


## XGBoost model:
train_x = data.matrix(train[, -7])
train_y = train[,7]

test_x = data.matrix(test[, -7])
test_y = test[, 7]

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

model4 = xgb.train(data = xgb_train, max.depth = 3, nrounds = 700, verbose = 0)

## Make predictions on the training data
train_pred <- predict(model4, xgb_train)

## Calculate R-squared on training data
train_r_squared <- caret::R2(pred = train_pred, obs = train_y)
print(paste("Training R-squared:", train_r_squared))

## Make predictions on the testing data
test_pred <- predict(model4, xgb_test)

## Calculate R-squared on testing data
test_r_squared <- caret::R2(pred = test_pred, obs = test_y)
print(paste("Testing R-squared:", test_r_squared))

importance_matrix = xgb.importance(colnames(xgb_train), model = model4)
xgb.plot.importance(importance_matrix[1:7,])

importance_matrix

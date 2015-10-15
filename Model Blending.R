################################################################################
#
# Perform Model blending
# For more info reger to http://mlwave.com/kaggle-ensembling-guide/
#
################################################################################

library(caret)
library(gbm)

### Pick your paramters
n_folds = 5
verbose = TRUE
shuffle = FALSE

### Make some fake data
N <- 1000
X1 <- runif(N);     X2 <- 2*runif(N); X3 <- 3*runif(N) 
Y  <- runif(N,0,1); Y[Y<0.5] <- 0;    Y[Y>=0.5] <- 1
data <- data.frame(y=Y,X1=X1,X2=X2,X3=X3)

### Just break it up to some train/test set
### X_submission and y_submission_actual represent the test set for Kaggle competitions
### In reality, we will never have y_submission_actual
X = data[,c("X1","X2","X3")];        y = data.frame(data[,c("y")])
X_submission = X[700:1000,];         X = X[1:700,]
y_submission_actual = y[700:1000,];  y = y[1:700,]


### Can introduce another round of randomness by shuffleing around the indices
if (shuffle){
  idx = sample(nrow(X))
  X = X[idx,]
  y = y[idx,] 
}

### Returns train inidices for n_folds using StratifiedKFold
skf = createFolds(y, k = n_folds , list = TRUE, returnTrain = TRUE)

### Create a list of models to run
clfs <- c("gbm1","gbm2")

### Pre-allocate the data
### For each model, add a column with N rows for each model
dataset_blend_train = matrix(0, nrow(X), length(clfs))
dataset_blend_test  = matrix(0, nrow(X_submission), length(clfs))

### Loop over the models
j <- 0 
for (clf in clfs){
  j <- j + 1
  print(paste(j,clf))
  
  ### Create a tempory array that is (Holdout_Size, N_Folds).
  ### Number of testing data x Number of folds , we will take the mean of the predictions later
  dataset_blend_test_j = matrix(0, nrow(X_submission), length(skf))
  print(paste(nrow(dataset_blend_test_j),ncol(dataset_blend_test_j)))
  
  ### Loop over the folds
  i <- 0
  for (sk in skf){
    i <- i + 1
    print(paste("Fold", i))
    
    ### Extract and fit the train/test section for each fold    
    tmp_train <- unlist(skf[i])
    X_train = X[tmp_train,]
    y_train = y[tmp_train]
    X_test  = X[-tmp_train,]
    y_test  = y[-tmp_train]
    
    ### Stupid hack to fit the model
    if (clf == "gbm1"){
      mod <- gbm(y_train~X1+X2+X3, data = data.frame(y_train,X_train), n.trees=1000, interaction.depth=3, train.fraction = 0.8)
      best.iter <- gbm.perf(mod,method="test", plot.it = FALSE)
      #best.iter <- gbm.perf(mod,method="cv")
      print(paste("Best iter:",best.iter))
    }
    else if (clf == "gbm2"){
      mod <- gbm(y_train~X1+X2+X3, data = data.frame(y_train,X_train), n.trees=1000, interaction.depth=8, train.fraction = 0.8)
      best.iter <- gbm.perf(mod,method="test", plot.it = FALSE)
      #best.iter <- gbm.perf(mod,method="cv")
      print(paste("Best iter:",best.iter))
    }
    
    ### Predict the probability of current folds test set and store results.
    ### This output will be the basis for our blended classifier to train against,
    ### which is also the output of our classifiers
    dataset_blend_train[-tmp_train, j] <- predict(mod, X_test, n.trees=best.iter, type="response")
    
    ### Predict the probabilty for the holdout set and store results
    dataset_blend_test_j[, i] <- predict(mod, X_submission, n.trees=best.iter, type="response")
  }
  
  ### Take mean of final holdout set folds
  dataset_blend_test[,j] = rowMeans(dataset_blend_test_j)
}


################################################################
#
# We can use original features + the meta features
#
################################################################

print ("Blending....")
mod <- gbm(y ~ X1 + X2 + X3 + X1.1 + X2.1, data = data.frame(y,X,dataset_blend_train), n.trees=1000, interaction.depth=8, train.fraction = 0.8)
y_submission = predict(mod, data.frame(X_submission,dataset_blend_test), type="response")




################################################################
#
# We can use only the meta features
#
################################################################


### We now have a new dataset with dimensions (N_train X N_models)
### Fit a logistic regression and predict on blended holdout set
print ("Blending....")
logit <- glm(y ~ X1 + X2, data = data.frame(y,dataset_blend_train), family = binomial(logit))
y_submission = predict(logit, data.frame(dataset_blend_test), type="response")


print ("Linear stretch of predictions to [0,1]")
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
y_submission <- range01(y_submission)

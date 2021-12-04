## Speed and Accuracy Test

library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)

set.seed(1)

trainn = 1000
testn = 1000
n = trainn + testn
p = 100
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
#y = 1 + X[, 1] + 2 * (X[, p/2+1] %in% c(1, 3)) + rnorm(n)


ntrees = 1000
ncores = 10
nmin = 20
mtry = p/2
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE 

######################################################
# Continuous Version
######################################################

y = 1 + X[, 1] + rnorm(n)
trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

xorder = order(testX[, 1])
testX = testX[xorder, ]
testY = testY[xorder]


seed1 <- runif(1,-1000, 1000)
seed2 <- runif(1,-1000, 1000)
seed_cont <- c(0,seed1, seed1, seed2, seed2 )

metric_cont = data.frame(matrix(NA, 5, 5))
rownames(metric_cont) = c("Noseed", "seed1_1", "seed1_2", "seed2_1", "seed2_2")
colnames(metric_cont) = c("fit.time", "pred.time", "pred.error", "obj.size", "ave.tree.size")

for (i in 1:5) {
  
  start_time <- Sys.time()
  RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry,
                split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, 
                importance = importance, seed = seed_cont[i])
  metric_cont[i, 1] = difftime(Sys.time(), start_time, units = "secs")
  start_time <- Sys.time()
  RLTPred <- predict(RLTfit, testX, ncores = ncores)
  metric_cont[i, 2] = difftime(Sys.time(), start_time, units = "secs")
  metric_cont[i, 3] = mean((RLTPred$Prediction - testY)^2)
  metric_cont[i, 4] = object.size(RLTfit)
  metric_cont[i, 5] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))
  
}


######################################################
# Categorical Version
######################################################

y = sample(seq(0,1), n, replace = TRUE)
  
trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

xorder = order(testX[, 1])
testX = testX[xorder, ]
testY = testY[xorder]


seed1 <- runif(1,-1000, 1000)
seed2 <- runif(1,-1000, 1000)
seed_cat <- c(0,seed1, seed1, seed2, seed2 )

metric_cat = data.frame(matrix(NA, 5, 5))
rownames(metric_cat) = c("Noseed", "seed1_1", "seed1_2", "seed2_1", "seed2_2")
colnames(metric_cat) = c("fit.time", "pred.time", "pred.error", "obj.size", "ave.tree.size")

for (i in 1:5) {
  
  start_time <- Sys.time()
  RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry,
                split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, 
                importance = importance, seed = seed_cat[i])
  metric_cat[i, 1] = difftime(Sys.time(), start_time, units = "secs")
  start_time <- Sys.time()
  RLTPred <- predict(RLTfit, testX, ncores = ncores)
  metric_cat[i, 2] = difftime(Sys.time(), start_time, units = "secs")
  metric_cat[i, 3] = mean((RLTPred$Prediction - testY)^2)
  metric_cat[i, 4] = object.size(RLTfit)
  metric_cat[i, 5] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))
  
}

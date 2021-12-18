## Speed and Accuracy Test

library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)

set.seed(1)

trainn = 1000
testn = 1000
n = trainn + testn
p = 1000
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
#y = 1 + X[, 1] + 2 * (X[, p/2+1] %in% c(1, 3)) + rnorm(n)
y = 1 + rowSums(X[, 1:(p/4)]) + rowSums(data.matrix(X[, (p/2) : (p/1.5)])) + rnorm(n)
#y = 1 + X[, 1] + rnorm(n)

ntrees = 100
ncores = 8
nmin = 2
mtry = p/2
sampleprob = 0.85
rule = "random"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE 

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

xorder = order(testX[, 1])
testX = testX[xorder, ]
testY = testY[xorder]

metric = data.frame(matrix(NA, 4, 5))
rownames(metric) = c("rlt", "rsf", "rf", "ranger")
colnames(metric) = c("fit.time", "pred.time", "pred.error", "obj.size", "ave.tree.size")

start_time <- Sys.time()
RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, nmin = nmin/2+1, mtry = mtry,
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, 
              importance = importance)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = mean((RLTPred$Prediction - testY)^2)
metric[1, 4] = object.size(RLTfit)
metric[1, 5] = mean(unlist(lapply(RLTfit$FittedForest$SplitVar, length)))

options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(y ~ ., data = data.frame(trainX, "y"= trainY), ntree = ntrees, nodesize = nmin, mtry = mtry, 
                nsplit = nsplit, sampsize = trainn*sampleprob, importance = importance)
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = mean((rsfpred$predicted - testY)^2)
metric[2, 4] = object.size(rsffit)
metric[2, 5] = rsffit$forest$totalNodeCount / rsffit$ntree

start_time <- Sys.time()
rf.fit <- randomForest(trainX, trainY, ntree = ntrees, mtry = mtry, nodesize = nmin, sampsize = trainn*sampleprob, importance = importance)
metric[3, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rf.pred <- predict(rf.fit, testX)
metric[3, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[3, 3] = mean((rf.pred - testY)^2)
metric[3, 4] = object.size(rf.fit)
metric[3, 5] = mean(colSums(rf.fit$forest$nodestatus != 0))

start_time <- Sys.time()
rangerfit <- ranger(trainY ~ ., data = data.frame(trainX), num.trees = ntrees, 
                    min.node.size = nmin, mtry = mtry, num.threads = ncores, 
                    sample.fraction = sampleprob, importance = "permutation",
                    respect.unordered.factors = "partition")
metric[4, 1] = difftime(Sys.time(), start_time, units = "secs")
rangerpred = predict(rangerfit, data.frame(testX))
metric[4, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[4, 3] = mean((rangerpred$predictions - testY)^2)
metric[4, 4] = object.size(rangerfit)
metric[4, 5] = mean(unlist(lapply(rangerfit$forest$split.varIDs, length)))

metric

par(mfrow=c(2,2))
par(mar = c(0.5, 2, 2, 2))

barplot(as.vector(RLTfit$VarImp), main = "RLT")
barplot(as.vector(rsffit$importance), main = "rsf")
barplot(rf.fit$importance[, 1], main = "rf")
barplot(as.vector(rangerfit$variable.importance), main = "ranger")


# multivariate split 

RLTfit <- RLT(trainX, trainY, ntrees = 1, ncores = 1, nmin = 100, 
              mtry = 3, linear.comb = 2)



# RLT split 


set.seed(1)

n = 1000
p = 100
X = matrix(rnorm(n*p), n, p)
y = 1 + X[, 1] + X[, 9] + X[, 3]  + rnorm(n)

testX = matrix(rnorm(n*p), n, p)
testy = 1 + testX[, 1] + testX[, 9] + testX[, 3]  + rnorm(n)


RLTfit <- RLT(X, y, ntrees = 1, ncores = 1, nmin = 100,
              mtry = 3, linear.comb = 1, reinforcement = TRUE,
              resample.prob = 0.8, resample.replace = TRUE,
              importance = TRUE, 
              param.control = list("embed.ntrees" = 100,
                                   "embed.mtry" = 4,
                                   "embed.nmin" = 10,
                                   "embed.split.gen" = "random",
                                   "embed.nsplit" = 3,
                                   "embed.resample.prob" = 0.75,
                                   "embed.mute" = 0.5))
get.one.tree(RLTfit, 1)

mean((RLTfit$OOBPrediction - y)^2)
pred = predict(RLTfit, testX) 
mean((pred$Prediction - testy)^2)


my_sample(1, 10, 10)


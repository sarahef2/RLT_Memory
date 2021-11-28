# Variance estimation of regression forest

library(RLT)

set.seed(1)

trainn = 1000
testn = 1000
n = trainn + testn
p = 20
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
y = 1 + X[, 1] + rnorm(n)

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]

xorder = order(testX[, 1])
testX = testX[xorder, ]
testY = testY[xorder]

## Variance Estimation Example

RLTfit <- RLT(trainX, trainY, ntrees = 20000, ncores = 10, nmin = 8, 
              mtry = p, split.gen = "random", nsplit = 3, resample.prob = 0.5, 
              resample.replace = FALSE, var.ready = TRUE)

RLTPred <- predict(RLTfit, testX, var.est = TRUE, ncores = 10, keep.all = TRUE)

mean(RLTPred$Variance < 0)

cover = (1 + testX$X1 > RLTPred$Prediction - 1.96*sqrt(RLTPred$Variance)) & 
  (1 + testX$X1 < RLTPred$Prediction + 1.96*sqrt(RLTPred$Variance))

mean(cover, na.rm = TRUE)

par(mfrow=c(1,1))
par(mar = rep(2, 4))
plot(RLTPred$Prediction, 1 + testX$X1,  pch = 19, cex = ifelse(is.na(cover), 1, 0.3), 
     col = ifelse(is.na(cover), "red", ifelse(cover, "green", "black")))
abline(0, 1, col = "red", lwd = 2)

## Variance Estimation Example (for k > n/2)


Var.Est = Reg_Var_Forest(trainX, trainY, testX, ncores = 10, nmin = 8,
                         mtry = p, split.gen = "random", nsplit = 3, 
                         ntrees = 50000, resample.prob = 0.75)

mean(Var.Est$var < 0)
alphalvl = 0.05

cover = (1 + testX$X1 > Var.Est$Prediction - qnorm(1-alphalvl/2)*sqrt(Var.Est$var)) & 
  (1 + testX$X1 < Var.Est$Prediction + qnorm(1-alphalvl/2)*sqrt(Var.Est$var))

mean(cover, na.rm = TRUE)

par(mfrow=c(1,1))
par(mar = rep(2, 4))
plot(Var.Est$Prediction, 1 + testX$X1,  pch = 19, cex = ifelse(is.na(cover), 1, 0.3), 
     col = ifelse(is.na(cover), "red", ifelse(cover, "green", "black")))
abline(0, 1, col = "red", lwd = 2)







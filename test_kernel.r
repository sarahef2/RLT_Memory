# Testing other features 

library(RLT)

set.seed(1)

trainn = 1000
testn = 1000
n = trainn + testn
p = 30
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


RLTfit <- RLT(trainX, trainY, ntrees = 1000, ncores = 10, nmin = 10, mtry = p/3,
              split.gen = "random", nsplit = 3, resample.prob = 0.8, 
              importance = TRUE)

# Obtain the tree structure of one tree

# getOneTree(RLTfit, 1)


# Forest Kernel

A = forest.kernel(RLTfit, testX)
heatmap(A$Kernel, Rowv = NA, Colv = NA, symm = TRUE)


RLTfit <- RLT(trainX, trainY, kernel.ready = TRUE)
RLTkernel = getKernelWeight(RLTfit, X[trainn + 1:2, ])
# heatmap(RLTkernel$Kernel[[1]], Rowv = NA, Colv = NA)

plot(trainX[, 1], trainX[, 2] + rnorm(trainn, sd = 0.1), pch = 19,
     cex = rowMeans(RLTkernel$Kernel[[1]])*15, xlab = "x1", ylab = "x2")

# peek a tree
getOneTree(RLTfit, 1)







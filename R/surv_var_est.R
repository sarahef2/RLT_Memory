#' @title                 Regression random forest with variance estimation
#' @description           Fit random forests with variance estimation at the 
#'                        given testing point. Only sampling without
#'                        replacement is available. 
#'                        
#' @param x               A `matrix` or `data.frame` of features
#' 
#' @param y               Response variable. a `numeric`/`factor` vector.
#'                        
#' @param testx           A `matrix` or `data.frame` of testing data
#'                        
#' @param ntrees          Number of trees, `ntrees = 100` if reinforcement is
#'                        used and `ntrees = 1000` otherwise.
#'                        
#' @param mtry            Number of randomly selected variables used at each 
#'                        internal node.
#'                        
#' @param nmin            Terminal node size. Splitting will stop when the 
#'                        internal node size is less than twice of `nmin`. This
#'                        is equivalent to setting `nodesize` = 2*`nmin` in the
#'                        `randomForest` package.
#'                        
#' @param alpha           Minimum number of observations required for each 
#'                        child node as a portion of the parent node. Must be 
#'                        within `[0, 0.5)`. When `alpha` $> 0$ and `split.gen`
#'                        is `rank` or `best`, this will force each child node 
#'                        to contain at least \eqn{\max(\texttt{nmin}, \alpha \times N_A)}
#'                        number of number of observations, where $N_A$ is the 
#'                        sample size at the current internal node. This is 
#'                        mainly for theoritical concern.  
#'                                              
#' @param k               Subsampling size.
#'                        
#' @param split.gen       How the cutting points are generated: `"random"`, 
#'                        `"rank"` or `"best"`. `"random"` performs random 
#'                        cutting point and does not take `alpha` into 
#'                        consideration. `"rank"` could be more effective when 
#'                        there are a large number of ties. It can also be used 
#'                        to guarantee child node size if `alpha` > 0. `"best"` 
#'                        finds the best cutting point, and can be cominbed with 
#'                        `alpha` too.
#' 
#' @param nsplit          Number of random cutting points to compare for each 
#'                        variable at an internal node.
#'                        
#' @param resample.prob   Proportion of in-bag samples.
#'                        
#' @param seed            Random seed using the `Xoshiro256+` generator.
#' 
#' @param ncores          Number of cores. Default is 1.
#' 
#' @param verbose         Whether fitting should be printed.
#' 
#' @param ...             Additional arguments.
#' 
#' @export
#' 
#' @return 
#' 
#' Prediction and variance estimation

surv_var_est <- function(x, y, censor, testx,
              			    ntrees = if (reinforcement) 100 else 500,
              			    mtry = max(1, as.integer(ncol(x)/3)),
              			    nmin = max(1, as.integer(log(nrow(x)))),
              			    alpha = 0,
              			    k = nrow(x) / 2, 
              			    split.gen = "random",
              			    nsplit = 1,
              			    seed = NaN,
              			    ncores = 1,
              			    type = "haz",
              			    verbose = 0,
              			    importance = FALSE,
              			    ...)
{
    if (!is.matrix(testx) & !is.data.frame(testx)) stop("testx must be a matrix or a data.frame")
    if (any(is.na(testx))) stop("NA not permitted in testx")
  
    # Look to switching to cumulative hazard or survival
  
    #Need to pre-specify observation track- make sure it is split nrow(X)/2 or smaller
    #Obstrack
    n <- dim(x)[1]
    tree_track1 <- vapply(1:(ntrees/2), 
                          function(i) sample(c(rep(1,k), rep(0,n-k)), n,
                                             replace = FALSE),
                          FUN.VALUE = numeric(n))
    tree_track2 <- 1-tree_track1
    tree_obtrack <- cbind(tree_track1, tree_track2)
    RLT.fit = RLT(x, y, censor, ntrees = ntrees, mtry = mtry, 
                nmin = nmin, alpha = alpha, nsplit = nsplit,
                split.gen = split.gen, #replacement = TRUE, 
                #resample.prob = k/n, 
                importance = importance, 
                  ncores = ncores,
                ObsTrack = tree_obtrack, track.obs = TRUE)
    
    RLT.pred = predict.RLT(RLT.fit, testx, ncores = ncores, keep.all = TRUE)
    
    # Moved to C++
    # tree.var = apply(RLT.pred$Allhazard[1,,], 2, var)
    
    # count how many pairs of trees match C
    
    # RLT.fit = RLT(x, y, censor, ntrees = ntrees, mtry = mtry, nmin = nmin, 
    #               alpha = alpha, nsplit = nsplit,
    #               split.gen = split.gen, replacement = FALSE, 
    #               resample.prob = k/n,
    #               ncores = ncores, track.obs = TRUE)
    # 
    # RLT.pred = predict.RLT(RLT.fit, testx, ncores = ncores, keep.all = TRUE)
    # Do the variance by time point and plot as a wiggly curve around hazard curve
    # Test with Cox P=5, 5 timepoints, get pointwise CI, plot the truth versus predicted
    
    #C_min = qhyper(0.01, k, n - k, k)
    #C_max = qhyper(0.99, k, n - k, k)
    
    #C = seq(C_min, C_max)
    C = seq(0, k-1);
    storage.mode(C) <- "integer"
    
    if(type=="haz"){
      pred <- RLT.pred$Allhazard
      #pred_tv <- RLT.pred_tv$Allhazard
    }else if(type=="cumhaz"){
      pred <- RLT.pred$Allhazard
      #pred_tv <- RLT.pred_tv$Allhazard
      for(i in 1:dim(pred)[3]){
        pred[,,i] <- apply(pred[,,i], 2, cumsum)
        #pred_tv[,,i] <- apply(pred_tv[,,i], 2, cumsum)
      }
    }else{
      pred <- RLT.pred$Allhazard
      #pred_tv <- RLT.pred_tv$Allhazard
      for(i in 1:dim(pred)[3]){
        pred[,,i] <- exp(-apply(pred[,,i], 2, cumsum))
        #pred_tv[,,i] <- exp(-apply(pred_tv[,,i], 2, cumsum))
      }
    }

    two.sample.var = EofVar_S(RLT.fit$ObsTrack, pred, 
                            pred, 
                            C, ncores, verbose)
    
    
    return(list("pred" = RLT.pred,
                "tree.var" = two.sample.var$tree.var,
                "tree.cov" = two.sample.var$tree.cov,
                "allc" = two.sample.var$allcounts,
                "estimation" = two.sample.var$estimation,
                "cov.estimation" = two.sample.var$cov.estimation,
                "var" = two.sample.var$var,
                "cov" = two.sample.var$cov))

}

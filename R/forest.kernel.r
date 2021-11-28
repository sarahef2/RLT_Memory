#' @title           kernel.RLT
#' 
#' @description     Get random forest induced kernel weight matrix of testing samples 
#'                  or between any two sets of data. This is an experimental feature.
#'                  Some components are not avaliable yet. 
#'                  
#' @param object    A fitted RLT object.
#' 
#' @param X1        The the first dataset. This calculates an \eqn{n_1 \times n_1} kernel
#'                  matrix of `X1`. 
#' 
#' @param X2        The the second dataset of the relative kernel weights are required. 
#'                  If \code{X2} is supplied, then calculate an \eqn{n_1 \times n_2} 
#'                  kernel matrix. If \code{vs.train} is used, then this must be the original 
#'                  training data.
#' 
#' @param vs.train  To calculate the kernel weights with respect to the training data. 
#'                  This is slightly different than supplying the training data to \code{X2}
#'                  due to re-samplings of the training process. Hence, \code{ObsTrack} must
#'                  be available from the fitted object (using \code{resample.track = TRUE}). 
#' 
#' @param ncores    Number of cores. Default is 1.
#' 
#' @param verbose   Whether fitting should be printed.
#' 
#' @param ... ...   Additional arguments.
#' @export

forest.kernel <- function(object,
                          X1 = NULL,
                          X2 = NULL,
                          vs.train = FALSE,
                          ncores = 1,
                          verbose = FALSE,
                          ...)
{
  if( class(object)[2] != "fit" )
    stop("object must be a fitted RLT object")

  if (is.null(X1))
    stop("self-kernel is not implemented yet.")

  if (!is.matrix(X1) & !is.data.frame(X1)) stop("X1 must be a matrix or a data.frame")
  {
    # check X1 data 
    if (is.null(colnames(X1))){
      if (ncol(X1) != object$parameters$p) 
        stop("X1 dimension does not match training data, variable names are not supplied...")
    }else if (any(colnames(X1) != object$xnames)){
      warning("X1 data variables names does not match training data ...")
    
      varmatch = match(object$xnames, colnames(X1))
    
      if (any(is.na(varmatch))) 
        stop("X1 is missing some variables from the orignal training data ...")
      
      X1 = X1[, varmatch]
    }
    
    X1 <- data.matrix(X1)
    
    if (is.null(X2))
    {
      K <- UniKernel_Self(object$FittedForest$SplitVar,
                          object$FittedForest$SplitValue,
                          object$FittedForest$LeftNode,
                          object$FittedForest$RightNode,
                          X1,
                          object$ncat,
                          ncores,
                          verbose)
    }else{

      # check X2
      
      if (!is.null(X2))
        if (!is.matrix(X2) & !is.data.frame(X2))
          stop("X2 must be a matrix or a data.frame")    
      
      if (is.null(colnames(X2))){
        if (ncol(X2) != object$parameters$p) 
          stop("X2 dimension does not match training data, variable names are not supplied...")
      }else if (any(colnames(X2) != object$xnames)){
        warning("X2 data variables names does not match training data ...")
        
        varmatch = match(object$xnames, colnames(X2))
        
        if (any(is.na(varmatch))) stop("X2 missing some variables from the orignal training data ...")
        
        X2 = X2[, varmatch]
      }
      
      if (!vs.train)
      {
        # cross-kernel of X1 and X2
        
        K <- UniKernel_Cross(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             X1,
                             X2,
                             object$ncat,
                             ncores,
                             verbose)

      }else{
        
        # kernel matrix as to the training process 
        # ObsTrack must be provided 
        if ( is.null(object$ObsTrack) )
          stop("Must have ObsTrack to perform vs.train")
      
        ObsTrack = object$ObsTrack
      
        if (nrow(ObsTrack) != nrow(X2))
          stop("X2 must be the original training data")
        
        K <- UniKernel_Train(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             X1,
                             X2,
                             ObsTrack,
                             object$ncat,
                             ncores,
                             verbose)
        
      }
    }
  }

  class(K) <- c("RLT", "kernel")
  
  return(K)
}

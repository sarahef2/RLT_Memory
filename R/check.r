#' @title check_input
#' @name check_input
#' @description Check input arguments to determine which model to use
#' @param x x
#' @param y x
#' @param censor censoring indicator
#' @param model model type
#' @keywords internal

check_input <- function(x, y, censor, model)
{
  if (!is.matrix(x) & !is.data.frame(x)) stop("x must be a matrix or a data.frame")
  if (!is.vector(y)) stop("y must be a vector")
  
  if (any(is.na(x))) stop("NA not permitted in x")
  if (any(is.na(y))) stop("NA not permitted in y")
  
  if (!is.null(y))
    if (nrow(x) != length(y)) stop("number of observations does not match: x & y")
  
  if (!is.numeric(y) & !is.factor(y))
    stop("y must be numeric or factor")
  
  if (!is.null(censor))
  {
    if (!is.vector(censor) | !is.numeric(censor)) stop("censor must be a numerical vector")
    if (length(y) != length(censor)) stop("number of observations does not match: y & censor")
    
    if ( sum(is.na(match(sort(unique(censor)), c(FALSE,TRUE)))) > 0 )
      stop("censoring indicator must be 0 (censored) or 1 (failed)")
    
    if ( all(censor == 0) )
      stop("all observations are censored, not suitable for survival model")
  }
  
  # decide which model to fit 
  
  if (is.null(model))
  {
    model = "regression" 
    
    if (!is.null(censor))
      model = "survival"
    
    if (is.factor(y))
      model = "classification"
    
    if ( is.numeric(y) & length(unique(y)) < 5 )
      warning("Number of unique values in y is less than 5. Please check input data and/or consider changing to classification.")
  }
  
  return(model)
}

#' @title check_param_RLT
#' @name check_param_RLT
#' @description Check parameters
#' 
#' \code{alpha} can specify a minimum number of observations required for each 
#' child node as a portion of the parent node. Must be within `[0, 0.5)`. When 
#' \code{alpha} $> 0$ and \code{split.gen} is `rank` or `best`, this will force 
#' each child node to contain at least \eqn{\max(\texttt{nmin}, \alpha \times N_A)}
#' number of number of observations, where $N_A$ is the sample size at the current 
#' internal node. This is mainly for theoretical concern. 
#' 
#' \code{split.rule} spcifies the splitting rule for comparisons. For regression, 
#' variance reduction `"var"` is used; for classification, `"gini"` index is used.
#' For survival, `"logrank"`, `"suplogrank"`, `"LL"` and `"penLL"` are available. When 
#' `"penLL"` is used, variable weights `"var.w"` are used as the penalty. 
#' 
#' @keywords internal

check_param_RLT <- function(n, p, ntrees, mtry, nmin,
                            split.gen, nsplit,
                            resample.replace, resample.prob,
                            resample.track,
                            use.obs.w, use.var.w,
                            importance,
                            var.ready, 
                            ncores, verbose,
                            reinforcement,
                            param.control)
{
  ntrees = max(ntrees, 1)
  storage.mode(ntrees) <- "integer"

  mtry = max(min(mtry, p), 1)
  storage.mode(mtry) <- "integer"  
  
  nmin = max(1, floor(nmin))
  storage.mode(nmin) <- "integer"
  
  # splitting rules 
  
  split.gen = match(split.gen, c("random", "rank", "best"))
  storage.mode(split.gen) <- "integer"  
  
  nsplit = max(1, nsplit)
  storage.mode(nsplit) <- "integer"  
  
  # resampling
  
  resample.replace = (resample.replace != 0)
  storage.mode(resample.replace) <- "integer"
  
  resample.prob = max(0, min(resample.prob, 1))
  storage.mode(resample.prob) <- "double"  
  
  resample.track = (resample.track != 0)
  storage.mode(resample.track) <- "integer"
  
  use.obs.w = (use.obs.w != 0)
  storage.mode(use.obs.w) <- "integer"
  
  use.var.w = (use.var.w != 0)
  storage.mode(use.var.w) <- "integer"

  # importance 
  
  importance = (importance != 0)
  storage.mode(importance) <- "integer"

  # variance estimation 
  
  var.ready = (var.ready != 0)
  storage.mode(var.ready) <- "integer"
  
  # system parameters
  ncores = max(ncores, 0)
  storage.mode(ncores) <- "integer"
  
  verbose = max(verbose, 0)
  storage.mode(verbose) <- "integer"

  # reinforcement learning settings
  
  reinforcement = (reinforcement != 0)
  storage.mode(reinforcement) <- "integer"

  # set RLT parameters
  
  if (!is.list(param.control)) {
    stop("param.control must be a list")
  }
  
  RLT.control = set_embed_param(param.control, reinforcement)

  param <- list("n" = n,
                "p" = p,
                "ntrees" = ntrees,
                "mtry" = mtry,
                "nmin" = nmin,
                "split.gen" = split.gen,
                "nsplit" = nsplit,
                "resample.replace" = resample.replace,
                "resample.prob" = resample.prob,
                "resample.track" = resample.track,
                "use.obs.w" = use.obs.w,
                "use.var.w" = use.var.w,
                "importance" = importance,
                "var.ready" = var.ready,
                "ncores" = ncores,
                "verbose" = verbose,                
                "reinforcement" = reinforcement)
  
  param = append(param, RLT.control)
  
  # additional parameters
  
  if (is.null(param.control$alpha)) {
    alpha <- 0
  } else alpha = min(max(alpha, 0), 0.5)
  storage.mode(alpha) <- "double"
  
  param$'alpha' = alpha
  param$'split.rule' = "var"
  param$'failcount' = 0
  param$'var.w.type' = 1
  
  # return
  return(param)
}


#' @title set_embed_param
#' @name set_embed_param
#' @description This is an internal function to set parameters for embedded model. 
#' @keywords internal

set_embed_param <- function(control, reinforcement)
{
  if (!reinforcement) ## no RLT, set some default to prevent crash
  {
    embed.ntrees = 1
    embed.resample.prob = 0.8
    embed.mtry.prop = 0.33
    embed.nmin = 1
    embed.split.gen = 1
    embed.nsplit = 1
  }else{

    if (is.null(control$embed.ntrees)) {
      embed.ntrees <- 50
    } else embed.ntrees = max(control$embed.ntrees, 1)
    
    storage.mode(embed.ntrees) <- "integer"
    
    if (is.null(control$embed.resample.prob)) {
      embed.resample.prob <- 0.8
    } else embed.resample.prob = max(0, min(control$embed.resample.prob, 1))
    
    storage.mode(embed.resample.prob) <- "double"
    
    if (is.null(control$embed.mtry.prop)) { # for embedded model, mtry is proportion
      embed.mtry.prop <- 1/2
    } else embed.mtry.prop = max(min(control$embed.mtry.prop, 1), 0)
    
    storage.mode(embed.mtry.prop) <- "double"
    
    if (is.null(control$embed.nmin)) {
      embed.nmin <- 5
    } else embed.nmin = max(1, floor(control$embed.nmin))
  
    storage.mode(embed.nmin) <- "double"
    
    if (is.null(control$embed.split.gen)) {
      embed.split.gen <- 1
    } else embed.split.gen = match(control$embed.split.gen, c("random", "rank", "best"))
    
    storage.mode(embed.split.gen) <- "integer"
    
    if (is.null(control$embed.nsplit)) {
      embed.nsplit <- 1
    } else embed.nsplit = max(1, control$embed.nsplit)
    
    storage.mode(embed.nsplit) <- "integer"
  }
  
  return(list("embed.ntrees" = embed.ntrees,
              "embed.resample.prob" = embed.resample.prob,
              "embed.mtry.prop" = embed.mtry.prop,
              "embed.nmin" = embed.nmin,
              "embed.split.gen" = embed.split.gen,
              "embed.nsplit" = embed.nsplit))
}
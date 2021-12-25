#' @title RegForest
#' @name RegForest
#' @description Internal function for fitting regression forest
#' @keywords internal

RegForest <- function(x, y, ncat,
                      obs.w, var.w,
                      resample.preset,
                      param,
                      ...)
{
  # prepare y
  storage.mode(y) <- "double"

  # check regression splitting rules
  all.split.rule = c("var")

  param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
  param$"split.rule" <- match(param$"split.rule", all.split.rule)
  
  if (param$verbose > 0)
    cat("Start fitting Regression Forest... \n")
  
  if (param$linear.comb == 1)
  {
    # fit model
    fit = RegUniForestFit(x, y, ncat,
                          obs.w, var.w,
                          resample.preset,
                          param)
  
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    
    class(fit) <- c("RLT", "fit", "reg", "uni")
  }else{
    # fit model
    fit = RegUniCombForestFit(x, y, ncat,
							  obs.w, var.w,
							  resample.preset,
							  param)
    
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    
    class(fit) <- c("RLT", "fit", "reg", "multi")
  }

  return(fit)
}

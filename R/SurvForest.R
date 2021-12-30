#' @title RegForest
#' @name RegForest
#' @description Internal function for fitting regression forest
#' @keywords internal

SurvForest <- function(x, y, censor, ncat,
                      obs.w, var.w,
                      resample.preset,
                      param,
                      ...)
{
  # prepare y
  storage.mode(y) <- "double"
  

  
  if (param$linear.comb == 1)
  {
    if (param$verbose > 0)
      cat("Fitting Survival Forest IN DEVELOPMENT... \n")    
      
    # check splitting rules
    all.split.rule = c("default", "logrank", "suplogrank")
    param$"split.rule" <- match.arg(param$"split.rule", all.split.rule)
    param$"split.rule" <- as.integer(match(param$"split.rule", all.split.rule))

    # fit single variable split model
    fit = SurvUniForestFit(x, y, censor, ncat,
                          obs.w, var.w,
                          resample.preset,
                          param)
  
    fit[["parameters"]] = param
    fit[["ncat"]] = ncat
    fit[["obs.w"]] = obs.w
    fit[["var.w"]] = var.w
    fit[["y"]] = y
    
    class(fit) <- c("RLT", "fit", "reg", "uni", "single")
  }else{
    cat("Linear combination fitting not implemented for survival random forests.")
  }

  return(fit)
}

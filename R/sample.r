#' @title                 samples
#' @description           testing function
#' @export mysample
my_sample <- function(Num, 
                      min, 
                      max)
{
  seed = runif(1, 0, 12312312)
  return( mysample(Num, min, max, seed) )
}

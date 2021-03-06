% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.r
\name{predict.RLT}
\alias{predict.RLT}
\title{prediction using RLT}
\usage{
\method{predict}{RLT}(
  object,
  testx = NULL,
  var.est = FALSE,
  keep.all = FALSE,
  ncores = 1,
  verbose = 0,
  ...
)
}
\arguments{
\item{object}{A fitted RLT object}

\item{testx}{The testing samples, must have the same structure as the
training samples}

\item{var.est}{Whether to estimate the variance of each testing data.
The original forest must be fitted with \code{var.ready = TRUE}.}

\item{keep.all}{whether to keep the prediction from all trees}

\item{ncores}{number of cores}

\item{...}{...}
}
\value{
A \code{RLT} prediction object, constructed as a list consisting

\item{Prediction}{Prediction}
\item{Variance}{if \code{var.est = TRUE} and the fitted object is
\code{var.ready = TRUE}}
}
\description{
Predict the outcome (regression, classification or survival)
using a fitted RLT object
}

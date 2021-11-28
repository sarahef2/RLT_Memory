//  **********************************
//  Reinforcement Learning Trees (RLT)
//  R bridging
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Tree_Functions.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
arma::umat ARMA_EMPTY_UMAT()
{
  arma::umat temp;
  return temp;
}

// [[Rcpp::export()]]
arma::vec ARMA_EMPTY_VEC()
{
  arma::vec temp;
  return temp;
}


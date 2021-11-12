//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Estimate the Expectation of Variance
//  **********************************

// my header file
# include "RLT.h"
# include "Utility//Trees.h"
# include "Utility/Utility.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List EofVar(arma::umat& ObsTrack,
            arma::mat& Pred,
            arma::uvec& C,
            int usecores,
            int verbose)
{
  DEBUG_Rcout << "-- calculate E(Var(Tree|C Shared)) ---" << std::endl;
  
  DEBUG_Rcout << C << std::endl;
  
  usecores = checkCores(usecores, verbose);
  
  size_t N = Pred.n_rows;
  size_t ntrees = Pred.n_cols;
  //The number of C's with which we will estimate the variance
  size_t length = C.n_elem;

  //For each observation, record the variance at each C  
  arma::mat Est(N, length, fill::zeros);
  //Keep track of the number of tree pairs used to calculate each C
  arma::uvec allcounts(length, fill::zeros);
   
#pragma omp parallel num_threads(usecores)
{
  #pragma omp for schedule(dynamic)
  for (size_t l = 0; l < length; l++) // calculate all C values
  {
    size_t count = 0;
    
    //For each pair of trees...
    for (size_t i = 0; i < (ntrees - 1); i++){
    for (size_t j = i+1; j < ntrees; j++){
      
      //Indices of the pair
      uvec pair = {i, j};
        
        //Pulls the columns related to the indices
        //Finds the minimum in each row
        //If the minimum is 1, then that observation was included in both rows
        //Count the number of obs used in both trees
        //If the sum of shared obs equals C(l)...
      if ( sum( min(ObsTrack.cols(pair), 1) ) == C(l) )
      {
        count++;
        
        //Calculate ..sigma_c and add it to the others
        Est.col(l) += 0.5 * square(Pred.col(i) - Pred.col(j));
      }
    }}
    
    //Take the mean of ..sigma_c
    Est.col(l) /= count;
    //Keep the count of ..sigma_c's
    allcounts(l) = count;
  }
  //We have now estimated \binom{n}{k}^{-2}sum(sum(..sigma_c))
}

  DEBUG_Rcout << "-- total count  ---" << allcounts << std::endl;  
  DEBUG_Rcout << "-- all estimates  ---" << Est << std::endl; 

  List ReturnList;
  
  ReturnList["allcounts"] = allcounts;
  ReturnList["estimation"] = Est;
  
  return(ReturnList);
}




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


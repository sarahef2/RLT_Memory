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
List EofVar(arma::mat& Pred,
            int usecores,
            int verbose)
{
  
  //Correction to TreeVar: see Tianning's paper

  usecores = checkCores(usecores, verbose);
  
  size_t N = Pred.n_rows;
  size_t ntrees = Pred.n_cols;
  size_t tree_pairs = Pred.n_cols/2;

  //For each observation, record the tree and sigmac sum  
  arma::vec Tree_Est(N, fill::zeros);
  arma::vec sigma_Est(N, fill::zeros);

  //Not sure if paralell would be useful here...
//#pragma omp parallel num_threads(usecores)
//{
  //#pragma omp for schedule(dynamic)
  for (size_t l = 0; l < tree_pairs; l++) // run through all indep. tree pairs
  {
    Tree_Est += 0.5*square(Pred.col(l) - Pred.col(l+tree_pairs));
  }

  Tree_Est/=tree_pairs;
  
  //count the number of pairs
  size_t count = 0;
  
  for(size_t i = 0; i < tree_pairs; i++){
    for(size_t j = i+1; j < tree_pairs; j++){
      
      count++;
      sigma_Est += 0.5 * square(Pred.col(i) - Pred.col(j));
    }
  }
  
  sigma_Est/=count;

  List ReturnList;
  
  ReturnList["tree_Est"] = Tree_Est;
  ReturnList["sigma_Est"] = sigma_Est;
  ReturnList["sigma"] = Tree_Est - sigma_Est;
  
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


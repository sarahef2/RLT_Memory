//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Utility.h"
# include "Reg_Uni/regForest.h"

using namespace Rcpp;
using namespace arma;

// Predict function- must be in the main source folder, 
//  otherwise Rcpp won't find it

// [[Rcpp::export()]]
List RegForestUniPred(arma::field<arma::ivec>& SplitVar,
          					  arma::field<arma::vec>& SplitValue,
          					  arma::field<arma::uvec>& LeftNode,
          					  arma::field<arma::uvec>& RightNode,
          					  arma::field<arma::vec>& NodeAve,
          					  arma::mat& X,
          					  arma::uvec& Ncat,
          					  arma::uvec& treeindex,
          					  bool keep_all,
          					  int usecores,
          					  int verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // convert R object to forest
  
  Reg_Uni_Forest_Class REG_FOREST(SplitVar, SplitValue, 
                                  LeftNode, RightNode, 
                                  NodeAve);

  // Initialize prediction objects  
  mat PredAll;
  
  // Run prediction
  Reg_Uni_Forest_Pred(PredAll,
                      (const Reg_Uni_Forest_Class&) REG_FOREST,
          					  X,
          					  Ncat,
          					  treeindex,
          					  usecores,
          					  verbose);
  
  // Initialize return list
  List ReturnList;

  ReturnList["Prediction"] = mean(PredAll, 1);

  // If keeping predictions for every tree  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
  
  return ReturnList;
}

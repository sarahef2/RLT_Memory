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
  //arma::field<arma::uvec>& NodeType,
  //arma::field<arma::vec>& NodeSize,
  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT REGRESSION ///" << std::endl;
  DEBUG_Rcout << "Check cores" << std::endl;
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // convert R object to forest
  
  Reg_Uni_Forest_Class REG_FOREST(//NodeType, 
                                  SplitVar, SplitValue, LeftNode, RightNode, //NodeSize, 
                                  NodeAve);
  
  mat PredAll;
  
  DEBUG_Rcout << "Start prediction" << std::endl;
  Reg_Uni_Forest_Pred(PredAll,
                      (const Reg_Uni_Forest_Class&) REG_FOREST,
          					  X,
          					  Ncat,
          					  treeindex,
          					  usecores,
          					  verbose);
  
  List ReturnList;

  ReturnList["Prediction"] = mean(PredAll, 1);
  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
  
  return ReturnList;
}

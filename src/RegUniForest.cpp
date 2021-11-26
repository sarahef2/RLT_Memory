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

// Fit function- must be in the main source folder, 
// otherwise Rcpp won't find it

// [[Rcpp::export()]]
List RegUniForestFit(arma::mat& X,
          					 arma::vec& Y,
          					 arma::uvec& Ncat,
          					 arma::vec& obsweight,
          					 arma::vec& varweight,
          					 arma::umat& ObsTrack,
          					 List& param)
{
  // reading parameters 
  PARAM_GLOBAL Param(param);

  // create data objects  
  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsweight, varweight);
  
  size_t N = REG_DATA.X.n_rows;
  size_t P = REG_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  int obs_track = Param.obs_track;

  int importance = Param.importance;

  // initiate forest argument objects
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);
  
  //Initiate forest object
  Reg_Uni_Forest_Class REG_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode, 
                                  NodeAve);
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // Initiate prediction objects
  vec Prediction;
  vec OOBPrediction;
  
  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  // Run model fitting
  Reg_Uni_Forest_Build((const RLT_REG_DATA&) REG_DATA,
                       REG_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       obs_id,
                       var_id,
                       ObsTrack,
                       Prediction,
                       OOBPrediction,
                       VarImp);

  //initialize return objects
  List ReturnList;
  
  List Forest_R;

  //Save forest objects as part of return list  
  Forest_R["SplitVar"] = SplitVar;
  Forest_R["SplitValue"] = SplitValue;
  Forest_R["LeftNode"] = LeftNode;
  Forest_R["RightNode"] = RightNode;
  Forest_R["NodeAve"] = NodeAve;
  
  //Add to return list
  ReturnList["FittedForest"] = Forest_R;
  
  if (obs_track) ReturnList["ObsTrack"] = ObsTrack;
  if (importance) ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = Prediction;
  ReturnList["OOBPrediction"] = OOBPrediction;

  return ReturnList;
}

// [[Rcpp::export()]]
List RegUniForestPred(arma::field<arma::ivec>& SplitVar,
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
  
  Reg_Uni_Forest_Class REG_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode, 
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

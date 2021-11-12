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
//  otherwise Rcpp won't find it

// [[Rcpp::export()]]
List RegForestUniFit(arma::mat& X,
          					 arma::vec& Y,
          					 arma::uvec& Ncat,
          					 List& param,
          					 List& RLTparam,
          					 arma::vec& obsweight,
          					 arma::vec& varweight,
          					 int usecores,
          					 int verbose,
          					 arma::umat& ObsTrack)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // reading parameters 
  PARAM_GLOBAL Param(param);
  if (verbose) Param.print();
  PARAM_RLT Param_RLT(RLTparam);
  if (verbose and Param.reinforcement) Param_RLT.print();

  // create data objects  
  RLT_REG_DATA REG_DATA(X, Y, Ncat, obsweight, varweight);
  
  size_t N = REG_DATA.X.n_rows;
  size_t P = REG_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  size_t seed = Param.seed;
  int obs_track = Param.obs_track;

  int importance = Param.importance;

  // initiate forest argument objects
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);
  
  //Initiate forest object
  Reg_Uni_Forest_Class REG_FOREST(SplitVar, SplitValue, 
                                  LeftNode, RightNode, 
                                  NodeAve);
  
  // VarImp
  vec VarImp;
  
  if (importance)
    VarImp.zeros(P);
  
  // Initiate prediction objects
  vec Prediction;
  vec OOBPrediction;
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // Run model fitting
  Reg_Uni_Forest_Build((const RLT_REG_DATA&) REG_DATA,
                       REG_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const PARAM_RLT&) Param_RLT,
                       obs_id,
                       var_id,
                       ObsTrack,
                       Prediction,
                       OOBPrediction,
                       VarImp,
                       seed,
                       usecores,
                       verbose);

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
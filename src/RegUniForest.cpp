//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Tree_Functions.h"
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
  
  if (obs_track) ReturnList["resample.track"] = ObsTrack;
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
                      bool VarEst,
                      bool keep_all,
                      size_t usecores,
                      size_t verbose)
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
                      usecores,
                      verbose);
  
  // Initialize return list
  List ReturnList;
  
  ReturnList["Prediction"] = mean(PredAll, 1);
  
  if (VarEst)
  {
    size_t nhalf = (size_t) REG_FOREST.SplitVarList.size()/2;
    
    uvec firsthalf = linspace<uvec>(0, nhalf-1, nhalf);
    uvec secondhalf = linspace<uvec>(nhalf, 2*nhalf-1, nhalf);
    
    vec SVar = var(PredAll, 0, 1); // norm_type = 1 means using n-1 as constant
    
    mat TreeDiff = PredAll.cols(firsthalf) - PredAll.cols(secondhalf);
    vec TreeVar = mean(square(TreeDiff), 1) / 2;
    
    vec Var = TreeVar*(1 + 1/2/nhalf) - SVar*(1 - 1/2/nhalf);

    ReturnList["Variance"] = Var;
  }
    
  
  // If keeping predictions for every tree  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
  
  return ReturnList;
}


// [[Rcpp::export()]]
List EofVar(arma::mat& Pred,
            int usecores,
            int verbose)
{
  
  usecores = checkCores(usecores, verbose);
  
  size_t N = Pred.n_rows;
  //size_t ntrees = Pred.n_cols;
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



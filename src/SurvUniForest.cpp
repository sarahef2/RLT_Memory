//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// Fit function- must be in the main source folder, 
// otherwise Rcpp won't find it

// [[Rcpp::export()]]
List SurvUniForestFit(arma::mat& X,
                     arma::uvec& Y,
                     arma::uvec& Censor,
                     arma::uvec& Ncat,
                     arma::vec& obsweight,
                     arma::vec& varweight,
                     arma::umat& ObsTrack,
                     List& param_r)
{
  // reading parameters 
  PARAM_GLOBAL Param;
  Param.PARAM_READ_R(param_r);
  
  if (Param.verbose) Param.print();
  
  size_t NFail = max( Y(find(Censor == 1)) );  
  
  // create data objects  
  RLT_SURV_DATA SURV_DATA(X, Y, Censor, Ncat, NFail, obsweight, varweight);
  
  size_t N = SURV_DATA.X.n_rows;
  size_t P = SURV_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  int obs_track = Param.obs_track;
  
  int importance = Param.importance;
  
  // initiate forest argument objects
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::field<arma::vec>> NodeHaz(ntrees);
  
  //Initiate forest object
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode, 
                                  NodeHaz);
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // Initiate prediction objects
  mat Prediction; // initialization means they will be calculated
  mat OOBPrediction;
  
  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  // Run model fitting
  Surv_Uni_Forest_Build((const RLT_SURV_DATA&) SURV_DATA,
                       SURV_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const uvec&) obs_id,
                       (const uvec&) var_id,
                       ObsTrack,
                       true,
                       Prediction,
                       OOBPrediction,
                       VarImp);
  
  //initialize return objects
  List ReturnList;
  ReturnList["NFail"] = NFail;
  
  List Forest_R;
  
  //Save forest objects as part of return list  
  Forest_R["SplitVar"] = SplitVar;
  Forest_R["SplitValue"] = SplitValue;
  Forest_R["LeftNode"] = LeftNode;
  Forest_R["RightNode"] = RightNode;
  Forest_R["NodeHaz"] = NodeHaz;
  
  //Add to return list
  ReturnList["FittedForest"] = Forest_R;
  
  if (obs_track) ReturnList["ObsTrack"] = ObsTrack;
  if (importance) ReturnList["VarImp"] = VarImp;
  
  ReturnList["Prediction"] = Prediction;
  ReturnList["OOBPrediction"] = OOBPrediction;
  
  
  // c index for model fitting 
  
  uvec nonNAs = find_finite(OOBPrediction.col(0));
  
  ReturnList["cindex"] = datum::nan;

  if (nonNAs.n_elem > 2)
  {
    vec oobpred(N, fill::zeros);
    
    for (auto i : nonNAs)
    {
      oobpred(i) = sum( cumsum( OOBPrediction.row(i) ) ); // sum of cumulative hazard as prediction
    }
    
    uvec oobY = Y(nonNAs);
    uvec oobC = Censor(nonNAs);
    vec oobP = oobpred(nonNAs);
    
    ReturnList["cindex"] =  cindex_i( oobY, oobC, oobP );
  }
  
  return ReturnList;
}

// [[Rcpp::export()]]
List SurvUniForestPred(arma::field<arma::ivec>& SplitVar,
                      arma::field<arma::vec>& SplitValue,
                      arma::field<arma::uvec>& LeftNode,
                      arma::field<arma::uvec>& RightNode,
                      arma::field<arma::field<arma::vec>>& NodeHaz,
                      arma::mat& X,
                      arma::uvec& Ncat,
                      size_t& NFail,
                      bool VarEst,
                      bool keep_all,
                      size_t usecores,
                      size_t verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);
  
  // convert R object to forest
  
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, 
                                  SplitValue, 
                                  LeftNode, 
                                  RightNode, 
                                  NodeHaz);
  
  // Initialize prediction objects  
  cube Pred;
  
  // Run prediction
  Surv_Uni_Forest_Pred(Pred,
                      (const Surv_Uni_Forest_Class&) SURV_FOREST,
                      X,
                      Ncat,
                      NFail,
                      usecores,
                      verbose);
  
  // Initialize return list
  List ReturnList;
  
  mat H(Pred.n_slices, Pred.n_rows);
  
#pragma omp parallel num_threads(usecores)
#pragma omp for schedule(static)
  for (size_t i = 0; i < Pred.n_slices; i++)
  {
    H.row(i) = mean(Pred.slice(i), 1).t();
  }
  
  ReturnList["hazard"] = H;  
  
  mat Surv(H);
  vec surv(H.n_rows, fill::ones);
  vec Ones(H.n_rows, fill::ones);
  
  for (size_t j=0; j < Surv.n_cols; j++)
  {
    surv = surv % (Ones - Surv.col(j)); //KM estimator
    Surv.col(j) = surv;
  }
  
  ReturnList["Survival"] = Surv;  
  
  if (keep_all)
    ReturnList["Allhazard"] = Pred;
  
  return ReturnList;
  }
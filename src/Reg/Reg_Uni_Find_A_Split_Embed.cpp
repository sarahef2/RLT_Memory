//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Find_A_Split_Embed(Uni_Split_Class& OneSplit,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                uvec& obs_id,
                                uvec& var_id,
                                Rand& rngl)
{
  
  Rcout << "    --- Reg_Uni_Find_A_Split_Embed " << std::endl;

  // set embeded model parameters 
  PARAM_GLOBAL Embed_Param;
  
  Embed_Param.N = obs_id.n_elem;
  Embed_Param.P = var_id.n_elem;
  Embed_Param.ntrees = Param.embed_ntrees;
  Embed_Param.nmin = Param.embed_nmin;
  Embed_Param.importance = 1;
  Embed_Param.seed = rngl.rand_sizet(0, INT_MAX);
  Embed_Param.ncores = 1;
  Embed_Param.verbose = 0;

  if (Param.embed_mtry > 1)
    Embed_Param.mtry = (size_t) Param.embed_mtry;
  else
    Embed_Param.mtry = (size_t) Embed_Param.P * Param.embed_mtry;
  
  Embed_Param.split_gen = Param.embed_split_gen;
  Embed_Param.nsplit = Param.embed_nsplit;
  Embed_Param.resample_prob = Param.embed_resample_prob;
  
  //Embed_Param.rlt_print();  
  size_t p_new;
  
  if (Param.embed_mute > 1)
    p_new = Embed_Param.P - Param.embed_mute;
  else
    p_new = Embed_Param.P - (size_t) Embed_Param.P * Param.embed_mute;

  if (p_new < 1)
    p_new = 1;
  
  // start fitting embedded model 
  
  size_t N = Embed_Param.N;
  size_t P = Embed_Param.P;
  size_t ntrees = Embed_Param.ntrees;

  umat ObsTrack;
    
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
  
  // Initiate prediction objects
  vec Prediction;
  vec OOBPrediction;
  
  // VarImp
  vec VarImp(P, fill::zeros);
  
  // Run model fitting
  
  Reg_Uni_Forest_Build(REG_DATA,
                       REG_FOREST,
                       (const PARAM_GLOBAL&) Embed_Param,
                       obs_id,
                       var_id,
                       ObsTrack,
                       Prediction,
                       OOBPrediction,
                       VarImp);
  
  Rcout << "variable importance is " << VarImp << std::endl;
  
  size_t var_best = var_id(VarImp.index_max());
  
  Rcout << "the best variable is " << var_best << std::endl;
  
  var_id = var_id(sort_index(VarImp, "descend"));
  
  Rcout << "variable order is" << var_id << std::endl;
  
  var_id.resize(p_new);
    
  Rcout << "after muting we will have " << p_new << " variates: " << var_id << std::endl;
  
  
  
  
}


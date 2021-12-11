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
  
  PARAM_GLOBAL Embed_Param;
  
  Embed_Param.print();
  Embed_Param.rlt_print();
  
  Embed_Param.N = obs_id.n_elem;
  Embed_Param.P = var_id.n_elem;

  Embed_Param.ntrees = Param.embed_ntrees;
  Embed_Param.nmin = Param.embed_nmin;
  
  if (Param.embed_mtry > 1)
    Embed_Param.mtry = (size_t) Param.embed_mtry;
  else
    Embed_Param.mtry = (size_t) Embed_Param.P * Param.embed_mtry;
  
  Embed_Param.split_gen = Param.embed_split_gen;
  Embed_Param.nsplit = Param.embed_nsplit;
  Embed_Param.resample_prob = Param.embed_resample_prob;
  
  Rcout << "after setting .... " << std::endl;
  
  
  Embed_Param.print();
  Embed_Param.rlt_print();  
  
  if (Param.embed_mute > 0)
    Rcout << "need to mute variates.. " << std::endl;

  
}


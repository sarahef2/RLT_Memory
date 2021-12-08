//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

void Reg_Multi_Find_A_Split_Embed(Multi_Split_Class& OneSplit,
                                  const RLT_REG_DATA& REG_DATA,
                                  const PARAM_GLOBAL& Param,
                                  uvec& obs_id,
                                  uvec& var_id,
                                  Rand& rngl)
{
  
  Rcout << "    --- Reg_Multi_Find_A_Split_Embed " << std::endl;
  
  PARAM_GLOBAL Embed_Param = Param;
  
  Embed_Param.print();
}


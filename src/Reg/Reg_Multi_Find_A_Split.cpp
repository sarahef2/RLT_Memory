//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Reg_Multi_Find_A_Split(Multi_Split_Class& OneSplit,
                            const RLT_REG_DATA& REG_DATA,
                            const PARAM_GLOBAL& Param,
                            uvec& obs_id,
                            uvec& var_id,
                            Rand& rngl)
{
  
  Rcout << "Reg_Multi_Find_A_Split:" << std::endl;
  
  uvec var_index = rngl.rand_uvec(Param.mtry, 0, var_id.n_elem - 1);
  
  // check categorical 

  
  // if all continuous
  uvec var_use = var_id(var_index);
  
  Rcout << "Sampled variables:\n" << var_use << std::endl;
  
  


  //If this variable is better than the last one tried
  OneSplit.var.zeros();
  OneSplit.load.zeros();
  OneSplit.load(0) = 1;
  
  OneSplit.value = 0.5;
  OneSplit.score = 1;
    
    
}
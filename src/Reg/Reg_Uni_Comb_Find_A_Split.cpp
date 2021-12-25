//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Reg_Uni_Comb_Find_A_Split(Comb_Split_Class& OneSplit,
                            const RLT_REG_DATA& REG_DATA,
                            const PARAM_GLOBAL& Param,
                            const uvec& obs_id,
                            const uvec& var_id,
                            Rand& rngl)
{
  
  Rcout << "Reg_Uni_Comb_Find_A_Split:" << std::endl;
  
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  //bool usevarweight = Param.usevarweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;

  // splitting variables
  uvec var_try = rngl.sample(var_id, mtry);

  // record splits
  uvec split_var(var_try.n_elem, fill::zeros);
  vec split_value(var_try.n_elem, fill::zeros);

  //If this variable is better than the last one tried
  OneSplit.var.zeros();
  OneSplit.load.zeros();
  OneSplit.load(0) = 1;
  
  OneSplit.value = 0.5;
  OneSplit.score = 1;
    
    
}
//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Utility//Trees.h"
# include "../Utility/Utility.h"
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

//Figuring out where to split a node, called from Split_A_Node
void Reg_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          uvec& var_id)
{
  
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  bool usevarweight = Param.usevarweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;
  size_t split_rule = Param.split_rule;
  bool reinforcement = Param.reinforcement;
  
  size_t N = obs_id.n_elem;
  size_t P = var_id.n_elem;
  
  mtry = ( (mtry <= P) ? mtry:P ); // take minimum

  // Choose the variables to try  
  uvec var_try = arma::randperm(P, mtry);
  
  //For each variable in var_try
  for (size_t j = 0; j < mtry; j++)
  {
    size_t temp_var = var_id(var_try(j));
    
    //Initialize objects
    Uni_Split_Class TempSplit;
    TempSplit.var = temp_var;
    TempSplit.value = 0;
    TempSplit.score = -1;
      
    if (REG_DATA.Ncat(temp_var) > 1) // categorical variable 
    {
      
      Reg_Uni_Split_Cat(TempSplit, 
                        obs_id, 
                        REG_DATA.X.unsafe_col(temp_var), 
                        REG_DATA.Ncat(temp_var),
                        REG_DATA.Y, 
                        REG_DATA.obsweight, 
                        0.0, // penalty
                        split_gen, 
                        split_rule, 
                        nsplit, 
                        nmin, 
                        alpha, 
                        useobsweight);

    }else{ // continuous variable
      
      Reg_Uni_Split_Cont(TempSplit,
                         obs_id,
                         REG_DATA.X.unsafe_col(temp_var), 
                         REG_DATA.Y,
                         REG_DATA.obsweight,
                         0.0, // penalty
                         split_gen,
                         split_rule,
                         nsplit,
                         nmin,
                         alpha,
                         useobsweight);
      
    }
    
    //If this variable is better than the last one tried
    if (TempSplit.score > OneSplit.score)
    {
      //Change to this variable
      OneSplit.var = TempSplit.var;
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
    }
  }
}
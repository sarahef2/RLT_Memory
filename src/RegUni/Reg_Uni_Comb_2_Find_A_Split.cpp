//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

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
  
  OneSplit.var.zeros();
  OneSplit.load.zeros();
  size_t linear_comb = OneSplit.var.n_elem;
  
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  bool useobsweight = Param.useobsweight;
  //bool usevarweight = Param.usevarweight;
  size_t nsplit = Param.nsplit;
  size_t split_gen = Param.split_gen;

  // splitting variables
  uvec var_try = rngl.sample(var_id, mtry);

  // record splits
  uvec split_var(var_try.n_elem, fill::zeros);
  vec split_score(var_try.n_elem, fill::zeros);
  //vec split_value(var_try.n_elem, fill::zeros);
  
  //explore each variable in var_try
  for (size_t j = 0; j < var_try.n_elem; j++)
  {
    size_t var_j = var_try(j);
    
    Rcout << "try variable " << var_j << std::endl;
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = var_j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    if (REG_DATA.Ncat(var_j) > 1) // categorical variable 
    {
      
      Reg_Uni_Split_Cat(TempSplit, 
                        obs_id, 
                        REG_DATA.X.unsafe_col(var_j), 
                        REG_DATA.Ncat(var_j),
                        REG_DATA.Y, 
                        REG_DATA.obsweight, 
                        0.0, // penalty
                        3, // best split
                        1, // splitting rule var (not used in function)
                        0, 
                        nmin, 
                        alpha, 
                        useobsweight,
                        rngl);
      
    }else{ // continuous variable
      
      Reg_Uni_Split_Cont(TempSplit,
                         obs_id,
                         REG_DATA.X.unsafe_col(var_j), 
                         REG_DATA.Y,
                         REG_DATA.obsweight,
                         0.0, // penalty
                         3, // best split
                         1, // splitting rule var (not used in function)
                         0,
                         nmin,
                         alpha,
                         useobsweight,
                         rngl);
      
    }
    
    split_var(j) = TempSplit.var;
    split_score(j) = TempSplit.score;    
    //split_value(j) = TempSplit.value;
    
  }
  
  uvec indices = sort_index(split_score, "descend");
  
  split_var = split_var(indices);
  split_score = split_score(indices);
  //split_value = split_value(indices);
  
  Rcout << "split_var \n" << split_var << std::endl;
  Rcout << "split_score \n" << split_score << std::endl;

  // if the best variable is categorical
  // do single categorical split
  // I may need to change this later for combination cat split
  if (REG_DATA.Ncat(split_var(0)) > 1)
  {
    size_t var_j = split_var(0);
    
    Rcout << "Use single cat split" <<  var_j << std::endl;
    
    //Initialize objects
    Split_Class TempSplit;
    TempSplit.var = var_j;
    TempSplit.value = 0;
    TempSplit.score = -1;
    
    Reg_Uni_Split_Cat(TempSplit, 
                      obs_id, 
                      REG_DATA.X.unsafe_col(var_j), 
                      REG_DATA.Ncat(var_j),
                      REG_DATA.Y, 
                      REG_DATA.obsweight, 
                      0.0, // penalty
                      split_gen,
                      1, // splitting rule var (not used in function)
                      nsplit, 
                      nmin, 
                      alpha, 
                      useobsweight,
                      rngl);
    
    OneSplit.var(0) = TempSplit.var;
    OneSplit.load(0) = 1;
    
    OneSplit.value = TempSplit.value;
    OneSplit.score = TempSplit.score;
    
    return;
  }else{
    
    // find all continuous variables at the top
    
    size_t cont_count = 0;
    uvec use_var(linear_comb, fill::zeros);
    
    for (size_t j = 0; j < linear_comb; j++)
      if (REG_DATA.Ncat(split_var(j)) == 1)
      {
        use_var(cont_count) = split_var(j);
        cont_count++;
      }

    if (cont_count == 1)
    {
      // use single variable split since only one good continuous variable
      size_t var_j = split_var(0);
      
      Rcout << "Use single cont split" <<  var_j << std::endl;
      
      //Initialize objects
      Split_Class TempSplit;
      TempSplit.var = var_j;
      TempSplit.value = 0;
      TempSplit.score = -1;
      
      Reg_Uni_Split_Cont(TempSplit,
                         obs_id,
                         REG_DATA.X.unsafe_col(var_j), 
                         REG_DATA.Y,
                         REG_DATA.obsweight,
                         0.0, // penalty
                         split_gen, // best split
                         1, // splitting rule var (not used in function)
                         nsplit,
                         nmin,
                         alpha,
                         useobsweight,
                         rngl);
      
      OneSplit.var(0) = TempSplit.var;
      OneSplit.load(0) = 1;
      
      OneSplit.value = TempSplit.value;
      OneSplit.score = TempSplit.score;
      
      return;

    }else{
      
      // continuous variable linear combination 
      // use all continuous variables in the top
      
      // there should be three types: sir (default), save, pca
      size_t split_rule = Param.split_rule;
      
      use_var.resize(cont_count); 
      
      Rcout << "Use comb cont split with variables \n" << use_var << std::endl;
      
      Rcout << "submatrix \n" << REG_DATA.X(obs_id, use_var) << std::endl;      
      
      // construct some new data 
      mat newX(REG_DATA.X(obs_id, use_var));
      vec newY(REG_DATA.Y(obs_id));
      vec newW;
      if (useobsweight) newW = REG_DATA.obsweight(obs_id);
      
      OneSplit.var = use_var;
      
      // find best linear combination split
      Reg_Uni_Comb_Split_Cont(OneSplit,
                              (const mat&) newX,
                              (const vec&) newY,
                              (const vec&) newW,
                              split_gen, // best split
                              split_rule,
                              nsplit,
                              nmin,
                              alpha,
                              useobsweight,
                              rngl);
    }
  }
  
  //If this variable is better than the last one tried
  OneSplit.var.zeros();
  OneSplit.load.zeros();
  OneSplit.load(0) = 0.5;
  OneSplit.load(1) = 0.5;
  
  OneSplit.value = 0.5;
  OneSplit.score = 0.3;
    
    
}
//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split using linear combination
void Reg_Uni_Comb_Split_Cont(Comb_Split_Class& OneSplit,
                             const uvec& use_var,
                             const RLT_REG_DATA& REG_DATA, 
                             const PARAM_GLOBAL& Param,
                             const uvec& obs_id,
                             Rand& rngl)
{
  
  Rcout << "Use comb cont split with variables \n" << use_var << std::endl;
 
  // construct some new data 
  bool useobsweight = Param.useobsweight;
  mat newX(REG_DATA.X(obs_id, use_var));
  vec newY(REG_DATA.Y(obs_id));
  vec newW;
  if (useobsweight) newW = REG_DATA.obsweight(obs_id);
  
  // some parameters
  // there are three split_rule types: sir (default), save, and pca
  size_t N = newX.n_rows;
  size_t split_rule = Param.split_rule;
  size_t split_gen = Param.split_gen;
  size_t nsplit = Param.nsplit;

  // find splitting rule 
  mat V;
  
  if (split_rule == 1) // default sir
  {
    Rcout << "using SIR split \n" << std::endl;
  }
  
  if (split_rule == 2) // save
  {
    Rcout << "using SAVE split \n" << std::endl;    
  }
  
  if (split_rule == 3) // pca
  {
    Rcout << "using PCA split \n" << std::endl;

    V = princomp(newX);
    
    Rcout << "PCA loadings \n" << V << std::endl;
  }
  
  // record splitting variable and loading
  OneSplit.var = use_var;
  OneSplit.load = V.col(0);
  
  // search for the best split
  arma::vec U1 = newX * V.col(0);
  //Rcout << "new linear combination x \n" << U1 << std::endl;
  
  if (split_gen == 1) // random split
  {
    Rcout << "random splitting \n" << std::endl;
    
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      double temp_cut = U1(rngl.rand_sizet(0,N-1)); 
      double temp_score = -1;
      
      // calculate score
      if (useobsweight)
        temp_score = reg_uni_cont_score_cut_full_w(U1, newY, temp_cut, newW);
      else
        temp_score = reg_uni_cont_score_cut_full(U1, newY, temp_cut);
      
      Rcout << "Try cut " << temp_cut << " with score " << temp_score << std::endl;
      
      if (temp_score > OneSplit.score)
      {
        OneSplit.score = temp_score;
        OneSplit.value = temp_cut;
      }
    }
    return;
  }

  // sort data 
  uvec indices = sort_index(U1);
  U1 = U1(indices);
  newY = newY(indices);
  if (useobsweight) newW = newW(indices);

  if (U1(0) == U1(U1.n_elem)) return;
  
  double alpha = Param.alpha;
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted
  // I need to do some changes to this
  // this will force nmin for each child node
  
  if (alpha > 0)
  {
    // if (N*alpha > nmin) nmin = (size_t) N*alpha;
    size_t nmin = (size_t) N*alpha;
    if (nmin < 1) nmin = 1;
    
    lowindex = nmin-1; // less equal goes to left
    highindex = N - nmin - 1;
  }
  
  // if ties
  // move index to better locations
  if ( U1(lowindex) == U1(lowindex+1) or U1(highindex) == U1(highindex+1) )
  {
    check_cont_index(lowindex, highindex, (const vec&) U1);
    
    if (lowindex > highindex)
    {
      Rcout << "lowindex > highindex... this shouldn't happen." << std::endl;
      return;
    }
  }
  
  if (split_gen == 2) // rank split
  {
    Rcout << "Rank split" << std::endl;
    return;
  }
  
  if (split_gen == 3) // best split
  {
    Rcout << "Best split" << std::endl;
    return;
  }

}

double reg_uni_cont_score_cut_full(const vec& x, 
                                   const vec& y, 
                                   double a_random_cut)
{
  size_t N = x.n_elem;
  
  double LeftSum = 0;
  double RightSum = 0;
  size_t LeftCount = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    //If x is less than the random cut, go left
    if ( x(i) <= a_random_cut )
    {
      LeftCount++;
      LeftSum += y(i);
    }else{
      RightSum += y(i);
    }
  }
  
  // if there are some observations in each node
  if (LeftCount > 0 && LeftCount < N)
    return LeftSum*LeftSum/LeftCount + RightSum*RightSum/(N - LeftCount);
  
  return -1;
}

double reg_uni_cont_score_cut_full_w(const vec& x, 
                                     const vec& y,
                                     double a_random_cut,
                                     const vec& w)
{
  size_t N = x.n_elem;
  
  double LeftSum = 0;
  double RightSum = 0;
  double Left_w = 0;
  double Right_w = 0;
  
  for (size_t i = 0; i < N; i++)
  {
    double wi = w(i);
    
    if ( x(i) <= a_random_cut )
    {
      Left_w += wi;
      LeftSum += y(i)*wi;
    }else{
      Right_w += wi;
      RightSum += y(i)*wi;
    }
  }
  
  if (Left_w > 0 && Right_w < N)
    return LeftSum*LeftSum/Left_w + RightSum*RightSum/Right_w;
  
  return -1;
  
}

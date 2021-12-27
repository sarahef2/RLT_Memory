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
                             const mat& newX,
                             const vec& newY,
                             const vec& newW,
                              int split_gen,
                              int split_rule,
                              int nsplit,
                              size_t nmin,
                              double alpha,
                              bool useobsweight,
                              Rand& rngl)
{
  size_t N = newX.n_rows;
  size_t P = newX.n_cols;
  
  if (split_rule == 1) // default sir
  {
    Rcout << "SIR splitting \n" << std::endl;
    
    
    return;
  }
  
  if (split_rule == 2) // save
  {
    
    Rcout << "SAVE splitting \n" << std::endl;    

    

    return;
  }
  
  if (split_rule == 3) // pca
  {
    
    Rcout << "PCA splitting \n" << std::endl;

    mat V = princomp(newX);
    mat U = newX * V;
    
    Rcout << "PCA loadings \n" << V << std::endl;
    Rcout << "PCA scores \n" << U << std::endl;
    
    for (size_t j = 0; j < P; j ++)
    {
      Rcout << "Try column " << j << std::endl;
      
      if (split_gen == 1) // random split
      {
        for (int k = 0; k < nsplit; k++)
        {
          // generate a random cut off
          double temp_cut = newX(rngl.rand_sizet(0,N-1), j); 
          double temp_score = -1;
          
          Rcout << "Try cut " << temp_cut << std::endl;
          
          // calculate score
          if (useobsweight)
            temp_score = reg_full_cont_score_at_cut_w(newX.unsafe_col(j), newY, temp_cut, newW);
          else
            temp_score = reg_full_cont_score_at_cut(newX.unsafe_col(j), newY, temp_cut);
          
          if (temp_score > OneSplit.score)
          {
            OneSplit.load = V.col(j);
            OneSplit.score = temp_score;
            OneSplit.value = temp_cut;
          }
        }
      }
      
      if (split_gen == 2) // rank split
      {
        Rcout << "Rank split" << std::endl;
      }
      
      if (split_gen == 3) // best split
      {
        Rcout << "Best split" << std::endl;
      }
      
    }
    
    return;
  }

}

double reg_full_cont_score_at_cut(const vec& xj, 
                                  const vec& y, 
                                  double temp_cut)
{
  return 0.5;
  
}

double reg_full_cont_score_at_cut_w(const vec& xj, 
                                    const vec& y,
                                    double temp_cut,
                                    const vec& w)
{
  return 0.5;
  
}

//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Forest_Pred(mat& Pred,
                         const Surv_Uni_Forest_Class& SURV_FOREST,
                  			 const mat& X,
                  			 const uvec& Ncat,
                  			 size_t usecores,
                  			 size_t verbose)
{
  size_t N = X.n_rows;
  size_t ntrees = SURV_FOREST.SplitVarList.size();
  
  Pred.zeros(N, ntrees);
  
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
        
      Surv_Uni_Tree_Class OneTree(SURV_FOREST.SplitVarList(nt),
                                 SURV_FOREST.SplitValueList(nt),
                                 SURV_FOREST.LeftNodeList(nt),
                                 SURV_FOREST.RightNodeList(nt),
                                 SURV_FOREST.NodeHazList(nt));
      
      Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      /*Pred.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);*/
    }
  }
}
//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees/Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Forest_Pred(cube& Pred,
                          const Surv_Uni_Forest_Class& SURV_FOREST,
                          const mat& X,
                          const uvec& Ncat,
                          size_t NFail,
                          const uvec& treeindex,
                          int usecores,
                          int verbose)
{
  DEBUG_Rcout << "/// Start prediction ///" << std::endl;
  
  size_t N = X.n_rows;
  size_t ntrees = SURV_FOREST.NodeTypeList.size();

  Pred.zeros(NFail + 1, treeindex.n_elem, N);

  //mat Pred(N, NFail + 1);
    
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < treeindex.n_elem; nt++)
    {
      
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      size_t whichtree = treeindex(nt);
      
      Surv_Uni_Tree_Class OneTree(SURV_FOREST.NodeTypeList(whichtree), 
                                  SURV_FOREST.SplitVarList(whichtree),
                                  SURV_FOREST.SplitValueList(whichtree),
                                  SURV_FOREST.LeftNodeList(whichtree),
                                  SURV_FOREST.RightNodeList(whichtree),
                                  //SURV_FOREST.NodeSizeList(whichtree),
                                  SURV_FOREST.NodeHazList(whichtree));

      Uni_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);

      vec oobpred(N, fill::zeros);
        
      for (size_t i = 0; i < N; i++)
      {
        Pred.slice(i).col(nt) = OneTree.NodeHaz(TermNode(i));
        oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) );
      }
    }
  }
  
  Pred.shed_row(0);
  
}


//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Utility//Trees.h"
# include "../Utility/Utility.h"
# include "regForest.h"

#include <xoshiro.h>
#include <dqrng_distribution.h>
#include <limits>
  
using namespace Rcpp;
using namespace arma;

void Reg_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          uvec& var_id,
                          umat& ObsTrack,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp)
{
  // parameters to use
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  size_t P = Param.P;
  size_t N = obs_id.n_elem;
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  bool importance = Param.importance;  
  size_t usecores = checkCores(Param.ncores, Param.verbose);
  size_t seed = Param.seed;

  // set seed
  Rand rng(seed);
  arma::uvec seed_vec = rng.rand_uvec(ntrees, 0, INT_MAX);
  
  // track obs matrix
  bool obs_track_pre = false; 
  
  if (ObsTrack.n_elem != 0) //if pre-defined
    obs_track_pre = true; 
  else
    ObsTrack.zeros(N, ntrees);
  
  // Calculate predictions
  Prediction.zeros(N);
  OOBPrediction.zeros(N);
  uvec oob_count(N, fill::zeros);
  
  // importance
  mat AllImp;
    
  if (importance)
    AllImp.zeros(ntrees, P);

  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(dynamic)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      // set xoshiro random seed
      Rand rngl(seed_vec(nt));
      
      // get inbag and oobag samples
      uvec inbagObs, oobagObs;
      
      //If ObsTrack isn't given, set ObsTrack
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement, rngl);
      
      // Find the samples
      get_samples(inbagObs, oobagObs, obs_id, ObsTrack.unsafe_col(nt));
      
      // initialize a tree (univariate split)
      Reg_Uni_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                 REG_FOREST.SplitValueList(nt),
                                 REG_FOREST.LeftNodeList(nt),
                                 REG_FOREST.RightNodeList(nt),
                                 REG_FOREST.NodeAveList(nt));
      
      size_t TreeLength = 1 + size/nmin*3;
      OneTree.initiate(TreeLength);

      // build the tree
      Reg_Uni_Split_A_Node(0, OneTree, REG_DATA, 
                           Param, inbagObs, var_id);
      
      // trim tree 
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      // inbag and oobag predictions for all subjects
      
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
    
      Uni_Find_Terminal_Node(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, obs_id, TermNode);
    
      Prediction += OneTree.NodeAve(TermNode);
      
      if (oobagObs.n_elem > 0)
      {
        OOBPrediction(oobagObs) += OneTree.NodeAve(TermNode(oobagObs));
        oob_count(oobagObs) += 1;
      }
      
      // calculate importance 
      
      if (importance and oobagObs.n_elem > 1)
      {
        uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));
        
        size_t NTest = oobagObs.n_elem;
        
        vec oobY = REG_DATA.Y(oobagObs);
        
        uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
        uvec TermNode(NTest, fill::zeros);
        
        Uni_Find_Terminal_Node(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, oobagObs, TermNode);
        
        vec oobpred = OneTree.NodeAve(TermNode);
        
        double baseImp = mean(square(oobY - oobpred));
        
        for (auto j : AllVar)
        {
          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
          uvec TermNode(NTest, fill::zeros);          
          
          vec tildex = shuffle( REG_DATA.X.unsafe_col(j).elem( oobagObs ) );
          
          Uni_Find_Terminal_Node_ShuffleJ(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, oobagObs, TermNode, tildex, j);
          
          // get prediction
          vec oobpred = OneTree.NodeAve(TermNode);

          // record
          AllImp(nt, j) =  mean(square(oobY - oobpred)) - baseImp;
        }
      }
    }
  }
  
  if (importance)
    VarImp = mean(AllImp, 0).t();
  
  Prediction /= ntrees;
  OOBPrediction = OOBPrediction / oob_count;
}
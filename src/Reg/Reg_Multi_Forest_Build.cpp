//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"
  
using namespace Rcpp;
using namespace arma;

void Reg_Multi_Forest_Build(const RLT_REG_DATA& REG_DATA,
                            Reg_Multi_Forest_Class& REG_FOREST,
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
  size_t linear_comb = Param.linear_comb;
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
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      // set xoshiro random seed
      Rand rngl(seed_vec(nt));
      
      // get inbag and oobag samples
      uvec inbagObs, oobagObs;
      
      //If ObsTrack isn't given, set ObsTrack
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement, rngl);

      // Rcout << ObsTrack.col(nt) << std::endl; 
      
      // Find the samples from pre-defined ObsTrack
      get_samples(inbagObs, oobagObs, obs_id, ObsTrack.unsafe_col(nt));
      
      // initialize a tree (multivariate split)      
      
      Reg_Multi_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                   REG_FOREST.SplitLoadList(nt),
                                   REG_FOREST.SplitValueList(nt),
                                   REG_FOREST.LeftNodeList(nt),
                                   REG_FOREST.RightNodeList(nt),
                                   REG_FOREST.NodeAveList(nt));      
      
      size_t TreeLength = 1 + size/nmin*3;
      OneTree.initiate(TreeLength, linear_comb);
      
      // build the tree
      Rcout << "build tree " << nt << std::endl;
      
      Reg_Multi_Split_A_Node(0, OneTree, REG_DATA, 
                             Param, inbagObs, var_id, rngl);
      
      // trim tree
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      Rcout << "print tree ..." << std::endl;
      
      OneTree.print();

    }
  }
  
}
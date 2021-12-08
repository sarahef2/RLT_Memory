//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Reg_Multi_Split_A_Node(size_t Node,
                            Reg_Multi_Tree_Class& OneTree,
                            const RLT_REG_DATA& REG_DATA,
                            const PARAM_GLOBAL& Param,
                            uvec& obs_id,
                            uvec& var_id,
                            Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;
  size_t linear_comb = Param.linear_comb;
  
  if (N < 2*nmin)
  {
    TERMINATENODE:
    Reg_Multi_Terminate_Node(Node, OneTree, obs_id, REG_DATA.Y, REG_DATA.obsweight, useobsweight);
    
  }else{
    
    //Set up another split
    uvec var(linear_comb, fill::zeros);
    vec load(linear_comb, fill::zeros);
    
    Multi_Split_Class OneSplit(var, load);
    
    //If reinforcement- NOT IMPLEMENTED
    if (Param.reinforcement)
    {
      Reg_Multi_Find_A_Split_Embed(OneSplit, REG_DATA, Param, obs_id, var_id, rngl);
    }else{
      //Figure out where to split the node
      Reg_Multi_Find_A_Split(OneSplit, REG_DATA, Param, obs_id, var_id, rngl);
    }
    
    OneSplit.print();
    
goto TERMINATENODE;
    
  }
}

// terminate and record a node

void Reg_Multi_Terminate_Node(size_t Node,
                              Reg_Multi_Tree_Class& OneTree,
                              uvec& obs_id,
                              const vec& Y,
                              const vec& obs_weight,
                              bool useobsweight)
{
  
  OneTree.SplitVar(Node, 0) = -1; // -1 says this node is a terminal node. Ow, it would be the variable num
  OneTree.LeftNode(Node) = obs_id.n_elem; // save node size on LeftNode
  
  //Find the average of the observations in the terminal node
  if (useobsweight)
  {
    double allweight = arma::sum(obs_weight(obs_id));
    OneTree.SplitValue(Node) = allweight; // save total weights on split value
    OneTree.NodeAve(Node) = arma::sum(Y(obs_id) % obs_weight(obs_id)) / allweight;
  }else{
    OneTree.NodeAve(Node) = arma::mean(Y(obs_id));
  }
}

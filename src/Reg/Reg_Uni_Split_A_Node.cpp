//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "regForest.h"

using namespace Rcpp;
using namespace arma;

//Split a node
void Reg_Uni_Split_A_Node(size_t Node,
                          Reg_Uni_Tree_Class& OneTree,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          uvec& var_id,
                          Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;

  // in rf, it is N <= nmin
  if (N < 2*nmin)
  {
    TERMINATENODE:
      Reg_Uni_Terminate_Node(Node, OneTree, obs_id, REG_DATA.Y, REG_DATA.obsweight, useobsweight);
    
  }else{
    
    //Set up another split
    Uni_Split_Class OneSplit;
    
    //If reinforcement- NOT IMPLEMENTED
    if (Param.reinforcement)
    {
      Reg_Uni_Find_A_Split_Embed(OneSplit, REG_DATA, Param, obs_id, var_id, rngl);
    }else{
      //Figure out where to split the node
      Reg_Uni_Find_A_Split(OneSplit, REG_DATA, Param, obs_id, var_id,rngl);
    }
    
    //Find the average for that node
    OneTree.NodeAve(Node) = arma::mean(REG_DATA.Y(obs_id));
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0)
      goto TERMINATENODE;
      
    // construct indices for left and right nodes

    uvec left_id(obs_id.n_elem);
    
    if ( REG_DATA.Ncat(OneSplit.var) == 1 )
    {
      split_id(REG_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id);  
      
    }else{
      split_id_cat(REG_DATA.X.unsafe_col(OneSplit.var), OneSplit.value, left_id, obs_id, REG_DATA.Ncat(OneSplit.var));
      
    }
    
    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == N or obs_id.n_elem == N)
      goto TERMINATENODE;
    
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    
    if ( OneTree.SplitVar( OneTree.SplitVar.size() - 2) >= 0 )
    {

      // extend tree structure
      OneTree.extend();
    }

    // get ready find the locations of next left and right nodes     
    size_t NextLeft = Node;
    size_t NextRight = Node;
    
    // record tree 
    
    OneTree.SplitVar(Node) = OneSplit.var;
    OneTree.SplitValue(Node) = OneSplit.value;

    //Find locations of the next nodes
    OneTree.find_next_nodes(NextLeft, NextRight);
    
    OneTree.LeftNode(Node) = NextLeft;
    OneTree.RightNode(Node) = NextRight;
    
    // split the left and right nodes 

    Reg_Uni_Split_A_Node(NextLeft, 
                         OneTree,
                         REG_DATA,
                         Param,
                         left_id, 
                         var_id,
                         rngl);

    
    Reg_Uni_Split_A_Node(NextRight,                          
                         OneTree,
                         REG_DATA,
                         Param,
                         obs_id, 
                         var_id,
                         rngl);

  }
}

// terminate and record a node

void Reg_Uni_Terminate_Node(size_t Node, 
                            Reg_Uni_Tree_Class& OneTree,
                            uvec& obs_id,                            
                            const vec& Y,
                            const vec& obs_weight,
                            bool useobsweight)
{
  
  OneTree.SplitVar(Node) = -1; // -1 says this node is a terminal node. Ow, it would be the variable num
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

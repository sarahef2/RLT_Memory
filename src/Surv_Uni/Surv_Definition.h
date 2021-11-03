//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

#define SurvWeightTH 1e-20

#ifndef SURV_DEFINITION
#define SURV_DEFINITION

// ************ //
//  Functions from Definition.h  //
// ************ //

// data
class RLT_SURV_DATA{
public:
  arma::mat& X;
  arma::uvec& Y;
  arma::uvec& Censor;
  arma::uvec& Ncat;
  size_t NFail;
  arma::vec& obsweight;
  arma::vec& varweight;
  
  RLT_SURV_DATA(arma::mat& X, 
                arma::uvec& Y,
                arma::uvec& Censor,
                arma::uvec& Ncat,
                size_t NFail,
                arma::vec& obsweight,
                arma::vec& varweight) : X(X), 
                Y(Y), 
                Censor(Censor),
                Ncat(Ncat),
                NFail(NFail),
                obsweight(obsweight), 
                varweight(varweight) {}
};

// Survival Forest Class

class Surv_Uni_Forest_Class{
public:
  arma::field<arma::uvec>& NodeTypeList;
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  //arma::field<arma::vec>& NodeSizeList;  
  arma::field<arma::field<arma::vec>>& NodeHazList;
  
  Surv_Uni_Forest_Class(arma::field<arma::uvec>& NodeTypeList,
                        arma::field<arma::ivec>& SplitVarList,
                        arma::field<arma::vec>& SplitValueList,
                        arma::field<arma::uvec>& LeftNodeList,
                        arma::field<arma::uvec>& RightNodeList,
                        //arma::field<arma::vec>& NodeSizeList,
                        arma::field<arma::field<arma::vec>>& NodeHazList) : NodeTypeList(NodeTypeList), 
                        SplitVarList(SplitVarList), 
                        SplitValueList(SplitValueList),
                        LeftNodeList(LeftNodeList),
                        RightNodeList(RightNodeList),
                        //NodeSizeList(NodeSizeList),
                        NodeHazList(NodeHazList) {}
};

// Survival Tree Class

class Surv_Uni_Tree_Class : public Uni_Tree_Class{
public:
  arma::field<arma::vec>& NodeHaz;
  
  Surv_Uni_Tree_Class(arma::uvec& NodeType,
                      arma::ivec& SplitVar,
                      arma::vec& SplitValue,
                      arma::uvec& LeftNode,
                      arma::uvec& RightNode,
                      //arma::vec& NodeSize,
                      arma::field<arma::vec>& NodeHaz) : Uni_Tree_Class(NodeType, 
                      SplitVar,
                      SplitValue,
                      LeftNode, 
                      RightNode//,
                                                                          //NodeSize
                      ),
                      NodeHaz(NodeHaz) {}
  
  
  // initiate tree
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0) TreeLength = 1;
    
    NodeType.zeros(TreeLength);
    
    SplitVar.set_size(TreeLength);
    SplitVar.fill(datum::nan);
    
    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    //NodeSize.zeros(TreeLength);    
    NodeHaz.set_size(TreeLength);
  }
  
  // trim tree 
  void trim(size_t TreeLength)
  {
    NodeType.resize(TreeLength);
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    //NodeSize.resize(TreeLength);    
    field_vec_resize(NodeHaz, TreeLength);
  }
  
  // extend tree 
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = NodeType.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);
    
    NodeType.resize(NewLength);
    NodeType(span(OldLength, NewLength-1)).zeros();
    
    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)).fill(datum::nan);
    
    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)).zeros();
    
    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)).zeros();
    
    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)).zeros();
    
    //NodeSize.resize(NewLength);
    //NodeSize(span(OldLength, NewLength-1)).zeros();    
    
    field_vec_resize(NodeHaz, NewLength);
  }
};


class Surv_Cat_Class: public Cat_Class{
public:
  arma::uvec FailCount;
  arma::uvec RiskCount;
  size_t nfail; 
  
  void initiate(size_t j, size_t NFail)
  {
    cat = j;
    nfail = 0;
    FailCount.zeros(NFail+1);
    RiskCount.zeros(NFail+1);
  }
  
  void print() {
    Rcout << "Category is " << cat << " weight is " << weight << " count is " << count << " data is\n" << 
      join_rows(FailCount, RiskCount) << std::endl;
  }
  
  void print_simple() {
    Rcout << "Category is " << cat << " weight is " << weight << " count is " << count << std::endl;
  }  
};


// ************ //
//  Functions from Tree.h  //
// ************ //

void move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Surv_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin);

double record_cat_split(std::vector<Surv_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);

double record_cat_split(size_t cat, 
                        std::vector<Surv_Cat_Class>& cat_reduced);

double record_cat_split(arma::uvec& goright_temp, 
                        std::vector<Surv_Cat_Class>& cat_reduced);                        


// other 

double cindex_d(arma::vec& Y,
                arma::uvec& Censor,
                arma::vec& pred);

double cindex_i(arma::uvec& Y,
                arma::uvec& Censor,
                arma::vec& pred);

#endif

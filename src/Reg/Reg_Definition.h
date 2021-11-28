//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

#ifndef RLT_REG_DEFINITION // include guard
#define RLT_REG_DEFINITION

// ************ //
//  data class  //
// ************ //

class RLT_REG_DATA{
public:
  arma::mat& X;
  arma::vec& Y;
  arma::uvec& Ncat;
  arma::vec& obsweight;
  arma::vec& varweight;
  
  RLT_REG_DATA(arma::mat& X, 
               arma::vec& Y,
               arma::uvec& Ncat,
               arma::vec& obsweight,
               arma::vec& varweight) : X(X), 
               Y(Y), 
               Ncat(Ncat), 
               obsweight(obsweight), 
               varweight(varweight) {}
};

// forest class regression 

class Reg_Uni_Forest_Class{
public:
  arma::field<arma::ivec>& SplitVarList;
  arma::field<arma::vec>& SplitValueList;
  arma::field<arma::uvec>& LeftNodeList;
  arma::field<arma::uvec>& RightNodeList;
  arma::field<arma::vec>& NodeAveList;
  
  Reg_Uni_Forest_Class(arma::field<arma::ivec>& SplitVarList,
                       arma::field<arma::vec>& SplitValueList,
                       arma::field<arma::uvec>& LeftNodeList,
                       arma::field<arma::uvec>& RightNodeList,
                       arma::field<arma::vec>& NodeAveList) : 
                       SplitVarList(SplitVarList), 
                       SplitValueList(SplitValueList),
                       LeftNodeList(LeftNodeList),
                       RightNodeList(RightNodeList),
                       NodeAveList(NodeAveList) {}
};

class Reg_Uni_Tree_Class : public Uni_Tree_Class{
public:
  arma::vec& NodeAve;

  Reg_Uni_Tree_Class(arma::ivec& SplitVar,
                     arma::vec& SplitValue,
                     arma::uvec& LeftNode,
                     arma::uvec& RightNode,
                     arma::vec& NodeAve) : Uni_Tree_Class(
                     SplitVar,
                     SplitValue,
                     LeftNode,
                     RightNode),
                     NodeAve(NodeAve) {}

  // initiate tree
  void initiate(size_t TreeLength)
  {
    if (TreeLength == 0) TreeLength = 1;

    SplitVar.set_size(TreeLength);
    SplitVar.fill(-2);

    SplitValue.zeros(TreeLength);
    LeftNode.zeros(TreeLength);
    RightNode.zeros(TreeLength);
    NodeAve.zeros(TreeLength);
  }

  // trim tree
  void trim(size_t TreeLength)
  {
    SplitVar.resize(TreeLength);
    SplitValue.resize(TreeLength);
    LeftNode.resize(TreeLength);
    RightNode.resize(TreeLength);
    NodeAve.resize(TreeLength);
  }

  // extend tree
  void extend()
  {
    // tree is not long enough, extend
    size_t OldLength = SplitVar.n_elem;
    size_t NewLength = (OldLength*1.5 > OldLength + 100)? (size_t) (OldLength*1.5):(OldLength + 100);

    SplitVar.resize(NewLength);
    SplitVar(span(OldLength, NewLength-1)).fill(datum::nan);

    SplitValue.resize(NewLength);
    SplitValue(span(OldLength, NewLength-1)).zeros();

    LeftNode.resize(NewLength);
    LeftNode(span(OldLength, NewLength-1)).zeros();

    RightNode.resize(NewLength);
    RightNode(span(OldLength, NewLength-1)).zeros();

    NodeAve.resize(NewLength);
    NodeAve(span(OldLength, NewLength-1)).zeros();
  }
};

class Reg_Cat_Class: public Cat_Class{
public:
  double y = 0;

  void calculate_score()
  {
    if (weight > 0)
      score = y / weight;
  }

  void print(void) {
    Rcout << "Category is " << cat << " count is " << count << " weight is " << weight << " y sum is " << y << " score is " << score << std::endl;
  }
};

//Move categorical index
void move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Reg_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin);

//Record category
double record_cat_split(std::vector<Reg_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);
#endif

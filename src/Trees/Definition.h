//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

#ifndef RLT_DEFINITION
#define RLT_DEFINITION

class PARAM_GLOBAL{
public:
  size_t N;
  size_t P;
  size_t ntrees;
  size_t mtry;
  size_t nmin;
  double alpha;
  int split_gen;
  int split_rule;
  int nsplit;
  bool replacement;
  double resample_prob;
  bool useobsweight;
  bool usevarweight;
  int varweighttype;
  int importance;  
  bool reinforcement;
  bool obs_track;
  size_t seed;
  bool failcount;
  
  PARAM_GLOBAL(List& param){
    N             = param["n"];
    P             = param["p"];
    ntrees        = param["ntrees"];
    mtry          = param["mtry"];
    nmin          = param["nmin"];
    alpha         = param["alpha"];
    split_gen     = param["split.gen"];
    split_rule    = param["split.rule"];
    nsplit        = param["nsplit"];
    replacement   = param["replacement"];
    resample_prob = param["resample.prob"];  
    useobsweight  = param["use.obs.w"];
    usevarweight  = param["use.var.w"];    
    varweighttype  = param["var.w.type"];    
    importance    = param["importance"];
    reinforcement = param["reinforcement"];
    obs_track     = param["track.obs"];
    seed          = param["seed"];
    failcount     = param["failcount"];
  }
  
  copyfrom(const PARAM_GLOBAL& Input){
      N             = Input.N;
      P             = Input.P;
      ntrees        = Input.ntrees;
      mtry          = Input.mtry;
      nmin          = Input.nmin;
      alpha         = Input.alpha;
      split_gen     = Input.split_gen;
      split_rule    = Input.split_rule;
      nsplit        = Input.nsplit;
      replacement   = Input.replacement;
      resample_prob = Input.resample_prob;
      useobsweight  = Input.useobsweight;
      usevarweight  = Input.usevarweight;
      importance    = Input.importance;
      reinforcement = Input.reinforcement;
      obs_track     = Input.obs_track;
      seed          = Input.seed;
  }
  
  void print() {
      Rcout << "--- Random Forest Parameters ---" << std::endl;
      Rcout << "            N = " << N << std::endl;
      Rcout << "            P = " << P << std::endl;
      Rcout << "       ntrees = " << ntrees << std::endl;
      Rcout << "         mtry = " << mtry << std::endl;
      Rcout << "         nmin = " << nmin << std::endl;
      Rcout << "        alpha = " << alpha << std::endl;
      Rcout << "    split_gen = " << ((split_gen == 1) ? "Random" : (split_gen == 2) ? "Rank" : "Best") << std::endl;
      if (split_gen < 3) Rcout << "   split_rule = " << split_rule << std::endl;
      Rcout << "       nsplit = " << nsplit << std::endl;
      Rcout << "  replacement = " << replacement << std::endl;
      Rcout << "resample prob = " << resample_prob << std::endl;
      Rcout << " useobsweight = " << (useobsweight ? "Yes" : "No") << std::endl;
      Rcout << " usevarweight = " << (usevarweight ? "Yes" : "No") << std::endl;
      Rcout << "   importance = " << (importance ? "Yes" : "No") << std::endl;
      Rcout << "reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
      Rcout << std::endl;
  }
};

class PARAM_RLT{
public:
    size_t embed_ntrees;
    double embed_resample_prob;
    double embed_mtry_prop;
    size_t embed_nmin;
    size_t embed_split_gen;
    size_t embed_nsplit;    
    
  PARAM_RLT(List& Param_RLT){
      embed_ntrees        = Param_RLT["embed.ntrees"];
      embed_resample_prob = Param_RLT["embed.resample.prob"];
      embed_mtry_prop     = Param_RLT["embed.mtry.prop"];
      embed_nmin          = Param_RLT["embed.nmin"];
      embed_split_gen     = Param_RLT["embed.split.gen"];
      embed_nsplit        = Param_RLT["embed.nsplit"];
  }
  
  void print() {
      Rcout << "--- Embedded Model Parameters ---" << std::endl;
      Rcout << "        embed_ntrees = " << embed_ntrees << std::endl;
      Rcout << " embed_resample_prob = " << embed_resample_prob << std::endl;
      Rcout << "     embed_mtry_prop = " << embed_mtry_prop << std::endl;
      Rcout << "          embed_nmin = " << embed_nmin << std::endl;
      Rcout << "     embed_split_gen = " << embed_split_gen << std::endl;
      Rcout << "        embed_nsplit = " << embed_nsplit << std::endl;
      Rcout << std::endl;
  }
  
};

// *************** //
// field functions //
// *************** //

void field_vec_resize(arma::field<arma::vec>& A, size_t size);
void field_vec_resize(arma::field<arma::uvec>& A, size_t size);


// *********************** //
//  Tree and forest class  //
// *********************** //

class Uni_Tree_Class{ // univariate split trees
public:
  //arma::uvec& NodeType;
  arma::ivec& SplitVar;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  //arma::vec& NodeSize;
  
  Uni_Tree_Class(//arma::uvec& NodeType,
                 arma::ivec& SplitVar,
                 arma::vec& SplitValue,
                 arma::uvec& LeftNode,
                 arma::uvec& RightNode//,
                 //arma::vec& NodeSize
                   ) : //NodeType(NodeType),
                                        SplitVar(SplitVar),
                                        SplitValue(SplitValue),
                                        LeftNode(LeftNode),
                                        RightNode(RightNode) {}//,
                                        //NodeSize(NodeSize) {}
  
  // find the next left and right nodes 
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    while( SplitVar(NextLeft)!=-2 ) NextLeft++;
    SplitVar(NextLeft) = -3;  
    
    NextRight = NextLeft;
    while( SplitVar(NextRight)!=-2 ) NextRight++;
    
    // -2: unused, -3: reserved; Else: internal node; -1: terminal node
    SplitVar(NextRight) = -3;
  }
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < SplitVar.n_elem and SplitVar(i) != -2) i++;
    return( (i < SplitVar.n_elem) ? i:SplitVar.n_elem );
  }
};


// **************** //
// class for splits //
// **************** //

class Uni_Split_Class{ // univariate splits
public:
  size_t var = 0;  
  double value = 0;
  double score = -1;
  
  void print(void) {
    Rcout << "Splitting varible is " << var << " value is " << value << " score is " << score << std::endl;
  }
};


// ************************ //
// for categorical variable //
// ************************ //


class Cat_Class{
public:
    size_t cat = 0;
    size_t count = 0; // count is used for setting nmin
    double weight = 0; // weight is used for calculation
    double score = 0; // for sorting
    
    void print() {
        Rcout << "Category is " << cat << " count is " << count << " weight is " << weight << " score is " << score << std::endl;
    }
};

#endif

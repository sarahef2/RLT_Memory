//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Tree Definitions
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
  size_t split_gen;
  size_t nsplit;
  bool replacement;
  double resample_prob;
  bool obs_track;
  bool useobsweight;
  bool usevarweight;
  size_t linear_comb;
  bool importance;
  bool reinforcement;
  size_t ncores;
  size_t verbose;
  size_t seed;
  bool failcount;
  double alpha;
  size_t split_rule;
  size_t varweighttype;
  
// RLT parameters 
  size_t embed_ntrees;
  double embed_resample_prob;
  double embed_mtry_prop;
  size_t embed_nmin;
  size_t embed_split_gen;
  size_t embed_nsplit;

  PARAM_GLOBAL(List& param){
    N             = param["n"];
    P             = param["p"];
    ntrees        = param["ntrees"];
    mtry          = param["mtry"];
    nmin          = param["nmin"];
    split_gen     = param["split.gen"];
    nsplit        = param["nsplit"];
    replacement   = param["resample.replace"];
    resample_prob = param["resample.prob"];
    obs_track     = param["resample.track"];
    useobsweight  = param["use.obs.w"];
    usevarweight  = param["use.var.w"];
    linear_comb   = param["linear.comb"];
    importance    = param["importance"];
    reinforcement = param["reinforcement"];
    ncores        = param["ncores"];
    verbose       = param["verbose"];
    seed          = param["seed"];
// other parameters
    alpha         = param["alpha"];
    failcount     = param["failcount"];
    varweighttype = param["var.w.type"];
    split_rule    = param["split.rule"];
// RLT parameters
    embed_ntrees        = param["embed.ntrees"];
    embed_resample_prob = param["embed.resample.prob"];
    embed_mtry_prop     = param["embed.mtry.prop"];
    embed_nmin          = param["embed.nmin"];
    embed_split_gen     = param["embed.split.gen"];
    embed_nsplit        = param["embed.nsplit"];
  };
  
  void copyfrom(const PARAM_GLOBAL& Input){
      N             = Input.N;
      P             = Input.P;
      ntrees        = Input.ntrees;
      mtry          = Input.mtry;
      nmin          = Input.nmin;
      split_gen     = Input.split_gen;
      nsplit        = Input.nsplit;
      replacement   = Input.replacement;
      resample_prob = Input.resample_prob;
      obs_track     = Input.obs_track;
      useobsweight  = Input.useobsweight;
      usevarweight  = Input.usevarweight;
      linear_comb   = Input.linear_comb;
      importance    = Input.importance;
      reinforcement = Input.reinforcement;
      ncores        = Input.ncores;
      verbose       = Input.verbose;
      seed          = Input.seed;
  // other parameters
      alpha         = Input.alpha;
      failcount     = Input.failcount;
      split_rule    = Input.split_rule;
      varweighttype = Input.varweighttype;
  // RLT parameters
      embed_ntrees        = Input.embed_ntrees;
      embed_resample_prob = Input.embed_resample_prob;
      embed_mtry_prop     = Input.embed_mtry_prop;
      embed_nmin          = Input.embed_nmin;
      embed_split_gen     = Input.embed_split_gen;
      embed_nsplit        = Input.embed_nsplit;
  };

  void print() {
      Rcout << "--- Random Forest Parameters ---" << std::endl;
  };
    
/*  
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
*/

};

// *************** //
// field functions //
// *************** //

void field_vec_resize(arma::field<arma::vec>& A, size_t size);
void field_vec_resize(arma::field<arma::uvec>& A, size_t size);
void field_vec_resize(arma::field<arma::ivec>& A, size_t size);

// *********************** //
//  Tree and forest class  //
// *********************** //

class Uni_Tree_Class{ // univariate split trees
public:
  arma::ivec& SplitVar;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  
  Uni_Tree_Class(arma::ivec& SplitVar,
                 arma::vec& SplitValue,
                 arma::uvec& LeftNode,
                 arma::uvec& RightNode) : SplitVar(SplitVar),
                                          SplitValue(SplitValue),
                                          LeftNode(LeftNode),
                                          RightNode(RightNode) {}
  
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

class Multi_Tree_Class{ // multivariate split trees
public:
  arma::imat& SplitVar;
  arma::mat& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  
  Multi_Tree_Class(arma::imat& SplitVar,
                   arma::mat& SplitValue,
                   arma::uvec& LeftNode,
                   arma::uvec& RightNode) : SplitVar(SplitVar),
                   SplitValue(SplitValue),
                   LeftNode(LeftNode),
                   RightNode(RightNode) {}
  
  void find_next_nodes(size_t& NextLeft, size_t& NextRight)
  {
    // -2: unused, -3: reserved; Else: internal node; -1: terminal node    
    
    while( SplitVar(NextLeft, 0) != -2 ) NextLeft++;
    SplitVar(NextLeft) = -3;  
    
    NextRight = NextLeft;
    
    while( SplitVar(NextRight, 0) != -2 ) NextRight++;
    SplitVar(NextRight, 0) = -3;
  }
  
  // get tree length
  size_t get_tree_length() {
    size_t i = 0;
    while (i < SplitVar.n_rows and SplitVar(i, 0) != -2) i++;
    return( (i < SplitVar.n_rows) ? i:SplitVar.n_rows );
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

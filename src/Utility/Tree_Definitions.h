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
  size_t N = 0;
  size_t P = 0;
  size_t ntrees = 1;
  size_t mtry = 1;
  size_t nmin = 1;
  size_t split_gen = 1;
  size_t nsplit = 1;
  bool replacement = 0;
  double resample_prob = 0.8;
  bool obs_track = 0;
  bool useobsweight = 0;
  bool usevarweight = 0;
  size_t linear_comb = 1;
  bool importance = 0;
  bool reinforcement = 0;
  size_t ncores = 1;
  size_t verbose = 0;
  size_t seed = 1;
// other parameters
  bool failcount = 0;
  double alpha = 0;
  size_t split_rule = 1;
  size_t varweighttype = 0;
// RLT parameters 
  size_t embed_ntrees = 0;
  double embed_resample_prob = 0;
  double embed_mtry = 0;
  size_t embed_nmin = 0;
  size_t embed_split_gen = 0;
  size_t embed_nsplit = 0;
  double embed_mute = 0;
  size_t embed_protect = 0;

  void PARAM_READ_R(List& param){
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
    embed_mtry          = param["embed.mtry"];
    embed_nmin          = param["embed.nmin"];
    embed_split_gen     = param["embed.split.gen"];
    embed_nsplit        = param["embed.nsplit"];
    embed_mute          = param["embed.mute"];
    embed_protect       = param["embed.protect"];
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
      embed_mtry          = Input.embed_mtry;
      embed_nmin          = Input.embed_nmin;
      embed_split_gen     = Input.embed_split_gen;
      embed_nsplit        = Input.embed_nsplit;
      embed_mute          = Input.embed_mute;
      embed_protect       = Input.embed_protect;
  };

  void print() {
      Rcout << "----- RLT Forest Parameters -----" << std::endl;
      Rcout << "         (N, P) = (" << N << ", " << P << ")"<< std::endl;
      Rcout << "         ntrees = " << ntrees << std::endl;
      Rcout << "           mtry = " << mtry << std::endl;
      Rcout << "           nmin = " << nmin << std::endl;
      Rcout << "      split_gen = " << ((split_gen == 1) ? "Random" : (split_gen == 2) ? "Rank" : "Best") << std::endl;
      if (split_gen < 3)
      Rcout << "         nsplit = " << nsplit << std::endl;
      Rcout << "    replacement = " << replacement << std::endl;
      Rcout << "  resample prob = " << resample_prob << std::endl;
      Rcout << "    Obs weights = " << (useobsweight ? "Yes" : "No") << std::endl;
      Rcout << "    Var weights = " << (usevarweight ? "Yes" : "No") << std::endl;
      Rcout << "  reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
      Rcout << std::endl;
  };
  
  void print() const {
    
    Rcout << "----- RLT Forest Parameters -----" << std::endl;
    Rcout << "         (N, P) = (" << N << ", " << P << ")"<< std::endl;
    Rcout << "         ntrees = " << ntrees << std::endl;
    Rcout << "           mtry = " << mtry << std::endl;
    Rcout << "           nmin = " << nmin << std::endl;
    Rcout << "      split_gen = " << ((split_gen == 1) ? "Random" : (split_gen == 2) ? "Rank" : "Best") << std::endl;
    if (split_gen < 3)
      Rcout << "         nsplit = " << nsplit << std::endl;
    Rcout << "    replacement = " << replacement << std::endl;
    Rcout << "  resample prob = " << resample_prob << std::endl;
    Rcout << "    Obs weights = " << (useobsweight ? "Yes" : "No") << std::endl;
    Rcout << "    Var weights = " << (usevarweight ? "Yes" : "No") << std::endl;
    Rcout << "  reinforcement = " << (reinforcement ? "Yes" : "No") << std::endl;
    Rcout << std::endl;    
    
  } 
  
  void rlt_print() {
    
    Rcout << "     embed.ntrees        = " << embed_ntrees << std::endl;
    Rcout << "     embed.resample_prob = " << embed_resample_prob << std::endl;
    Rcout << "     embed.mtry          = " << embed_mtry << std::endl;
    Rcout << "     embed.nmin          = " << embed_nmin << std::endl;
    Rcout << "     embed.split_gen     = " << embed_split_gen << std::endl;
    Rcout << "     embed.nsplit        = " << embed_nsplit << std::endl;
    Rcout << "     embed.mute          = " << embed_mute << std::endl;
    Rcout << "     embed.protect       = " << embed_protect << std::endl;
    
  };
  
  void rlt_print() const {
    
    Rcout << "     embed_ntrees        = " << embed_ntrees << std::endl;
    Rcout << "     embed_resample_prob = " << embed_resample_prob << std::endl;
    Rcout << "     embed_mtry          = " << embed_mtry << std::endl;
    Rcout << "     embed_nmin          = " << embed_nmin << std::endl;
    Rcout << "     embed_split_gen     = " << embed_split_gen << std::endl;
    Rcout << "     embed_nsplit        = " << embed_nsplit << std::endl;
    Rcout << "     embed_mute          = " << embed_mute << std::endl;
    Rcout << "     embed.protect       = " << embed_protect << std::endl;
    
  };
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

class Tree_Class{ // single split trees
public:
  arma::ivec& SplitVar;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  
  Tree_Class(arma::ivec& SplitVar,
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
  
  void print() {
    
    Rcout << "This tree has length " << get_tree_length() << std::endl;
    
  }
  
};

class Comb_Tree_Class{ // multivariate split trees
public:
  arma::imat& SplitVar;
  arma::mat& SplitLoad;
  arma::vec& SplitValue;
  arma::uvec& LeftNode;
  arma::uvec& RightNode;
  
  Comb_Tree_Class(arma::imat& SplitVar,
                   arma::mat& SplitLoad,
                   arma::vec& SplitValue,
                   arma::uvec& LeftNode,
                   arma::uvec& RightNode) : 
                   SplitVar(SplitVar),
                   SplitLoad(SplitLoad),
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
  
  void print() {
    
    Rcout << "This tree has length " << get_tree_length() << std::endl;

  }
  
};


// **************** //
// class for splits //
// **************** //

class Split_Class{ // univariate splits
public:
  size_t var = 0;  
  double value = 0;
  double score = -1;
  
  void print(void) {
    Rcout << "Splitting varible is " << var << " value is " << value << " score is " << score << std::endl;
  }
};

class Comb_Split_Class{ // multi-variate splits
public:
  arma::uvec& var;
  arma::vec& load;
  double value = 0;
  double score = -1;
  
  Comb_Split_Class(arma::uvec& var,
                   arma::vec& load) : 
    var(var),
    load(load) {}
  
  void print(void) {
    Rcout << "Splitting varible is\n" << var << "load is\n" << load << "value is " << value << "; score is " << score << std::endl;
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

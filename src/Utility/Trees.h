//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

# include <RcppArmadillo.h>
# include <Rcpp.h>
# include <algorithm>

using namespace Rcpp;
using namespace arma;

# include "Trees_Definition.h"
# include "Utility.h"

// ********************//
// functions for trees //
// ********************//

#ifndef RLT_ARRANGE
#define RLT_ARRANGE

void Uni_Find_Terminal_Node(size_t Node, 
              							const Uni_Tree_Class& OneTree,
              							const mat& X,
              							const uvec& Ncat,
              							uvec& proxy_id,
              							const uvec& real_id,
              							uvec& TermNode);

void Uni_Find_Terminal_Node_ShuffleJ(size_t Node, 
                                     const Uni_Tree_Class& OneTree,
                                     const mat& X,
                                     const uvec& Ncat,
                                     uvec& proxy_id,
                                     const uvec& real_id,
                                     uvec& TermNode,
                                     const vec& tildex,
                                     const size_t j);

List ForestKernelUni(arma::field<arma::uvec>& NodeType,
                     arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::field<arma::field<arma::uvec>>& NodeRegi,
                     arma::mat& X,
                     arma::uvec& Ncat,
                     arma::vec& obsweight,
                     int usecores,
                     int verbose);


// ************************//
// miscellaneous functions //
// ************************//

// catigorical variables pack
double pack(const size_t nBits, const uvec& bits);
void unpack(const double pack, const size_t nBits, uvec& bits);
bool unpack_goright(double pack, const size_t cat);

// sample both inbag and oobag samples
void oob_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const size_t size,
                 const bool replacement);

void set_obstrack(arma::umat& ObsTrack,
                  const size_t nt,
                  const size_t size,
                  const bool replacement,
                  Rand& rngl);

void get_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const arma::uvec& ObsTrack_nt);


void move_cont_index(size_t& lowindex, size_t& highindex, const vec& x, const uvec& indices, size_t nmin);
void split_id(const vec& x, double value, uvec& left_id, uvec& obs_id);
void split_id_cat(const vec& x, double value, uvec& left_id, uvec& obs_id, size_t ncat);


bool cat_reduced_compare(Cat_Class& a, Cat_Class& b);
bool cat_reduced_collapse(Cat_Class& a, Cat_Class& b); 

void goright_roller(arma::uvec& goright_cat);

#endif












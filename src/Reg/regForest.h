//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression Functions
//  **********************************

// my header file
# include "../RLT.h"
# include "../Utility/Tree_Functions.h"
# include "../Utility/Utility.h"
# include "Reg_Definition.h"

using namespace Rcpp;
using namespace arma;

#ifndef RLT_RegForest
#define RLT_RegForest

// univariate tree split functions 

List RegUniForestFit(mat& X,
          	         vec& Y,
          		       uvec& Ncat,
          		       vec& obsweight,
          		       vec& varweight,
          		       umat& ObsTrackPre,
          		       List& param);

void Reg_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          uvec& var_id,
                          umat& ObsTrack,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp);

void Reg_Uni_Split_A_Node(size_t Node,
                          Reg_Uni_Tree_Class& OneTree,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          uvec& var_id,
                          Rand& rngl);

void Reg_Uni_Terminate_Node(size_t Node, 
                            Reg_Uni_Tree_Class& OneTree,
                            uvec& obs_id,
                            const vec& Y,
                            const vec& obs_weight,
                            bool useobsweight);


void Reg_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                          const RLT_REG_DATA& REG_DATA,
                          const PARAM_GLOBAL& Param,
                          uvec& obs_id,
                          uvec& var_id,
                          Rand& rngl);

void Reg_Uni_Find_A_Split_Embed(Uni_Split_Class& OneSplit,
                                const RLT_REG_DATA& REG_DATA,
                                const PARAM_GLOBAL& Param,
                                uvec& obs_id,
                                uvec& var_id,
                                Rand& rngl);

void Reg_Uni_Split_Cont(Uni_Split_Class& TempSplit,
                        uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        const vec& obs_weight,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl);

void Reg_Uni_Split_Cat(Uni_Split_Class& TempSplit,
                       uvec& obs_id,
                       const vec& x,
                       const size_t ncat,
                       const vec& Y,
                       const vec& obs_weight,
                       double penalty,
                       int split_gen,
                       int split_rule,
                       int nsplit,
                       size_t nmin,
                       double alpha,
                       bool useobsweight,
                       Rand& rngl);

// splitting score calculations (continuous)

double reg_cont_score_at_cut(uvec& obs_id,
                            const vec& x,
                            const vec& Y,
                            double a_random_cut);

double reg_cont_score_at_cut_w(uvec& obs_id,
                              const vec& x,
                              const vec& Y,
                              double a_random_cut,
                              const vec& obs_weight);

double reg_cont_score_at_index(uvec& indices,
                              const vec& Y,
                              size_t a_random_ind);

double reg_cont_score_at_index_w(uvec& indices,
                                const vec& Y,
                                size_t a_random_ind,
                                const vec& obs_weight);

void reg_cont_score_best(uvec& indices,
                        const vec& x,
                        const vec& Y,
                        size_t lowindex, 
                        size_t highindex, 
                        double& temp_cut, 
                        double& temp_score);


void reg_cont_score_best_w(uvec& indices,
                          const vec& x,
                          const vec& Y,
                          size_t lowindex, 
                          size_t highindex, 
                          double& temp_cut, 
                          double& temp_score,
                          const vec& obs_weight);

// splitting score calculations (categorical)

double reg_cat_score(std::vector<Reg_Cat_Class>& cat_reduced, 
                     size_t temp_cat, 
                     size_t true_cat);

double reg_cat_score_w(std::vector<Reg_Cat_Class>& cat_reduced, 
                       size_t temp_cat, 
                       size_t true_cat);

void reg_cat_score_best(std::vector<Reg_Cat_Class>& cat_reduced, 
                        size_t lowindex,
                        size_t highindex,
                        size_t true_cat,
                        size_t& best_cat,
                        double& best_score);

void reg_cat_score_best_w(std::vector<Reg_Cat_Class>& cat_reduced, 
                          size_t lowindex,
                          size_t highindex,
                          size_t true_cat,
                          size_t& best_cat,
                          double& best_score);

// for prediction 

void Reg_Uni_Forest_Pred(mat& Pred,
                         const Reg_Uni_Forest_Class& REG_FOREST,
                         const mat& X,
                         const uvec& Ncat,
                         size_t usecores,
                         size_t verbose);

#endif

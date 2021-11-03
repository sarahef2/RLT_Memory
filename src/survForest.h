//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility//Utility.h"
# include "Surv_Uni//Surv_Definition.h"

using namespace Rcpp;
using namespace arma;

#ifndef RegForest_Fun
#define RegForest_Fun

// univariate tree split functions 

List SurvForestUniFit(arma::mat& X,
                      arma::uvec& Y,
                      arma::uvec& Censor,
                      arma::uvec& Ncat,
                      List& param,
                      List& RLTparam,
                      arma::vec& obsweight,
                      arma::vec& varweight,
                      int usecores,
                      int verbose,
                      arma::umat& ObsTrackPre);

void Surv_Uni_Forest_Build(const RLT_SURV_DATA& REG_DATA,
                           Surv_Uni_Forest_Class& SURV_FOREST,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           uvec& obs_id,
                           uvec& var_id,
                           umat& ObsTrack,
                           mat& Prediction,
                           mat& OOBPrediction,
                           vec& VarImp,
                           size_t seed,
                           int usecores,
                           int verbose);

void Surv_Uni_Split_A_Node(size_t Node,
                           Surv_Uni_Tree_Class& OneTree,
                           const RLT_SURV_DATA& SURV_DATA,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           uvec& obs_id,
                           uvec& var_id);

void Surv_Uni_Terminate_Node(size_t Node, 
                             Surv_Uni_Tree_Class& OneTree,
                             uvec& obs_id,
                             const uvec& Y,
                             const uvec& Censor,
                             const size_t NFail,
                             const vec& obs_weight,
                             const PARAM_GLOBAL& Param,
                             bool useobsweight);

void Surv_Uni_Find_A_Split(Uni_Split_Class& OneSplit,
                           const RLT_SURV_DATA& SURV_DATA,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& RLTParam,
                           uvec& obs_id,
                           uvec& var_id);

void Surv_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y, // Y is collapsed
                         const uvec& Censor, // Censor is collapsed
                         size_t NFail,
                         const uvec& All_Fail,
                         const vec& All_Risk,
                         vec& Temp_Vec,
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         bool failforce);

void Surv_Uni_Split_Cont_W(Uni_Split_Class& TempSplit, 
                           uvec& obs_id,
                           const vec& x,
                           const uvec& Y, // Y is collapsed
                           const uvec& Censor, // Censor is collapsed
                           const vec& obs_weight,
                           size_t NFail,
                           double penalty,
                           int split_gen,
                           int split_rule,
                           int nsplit,
                           size_t nmin, 
                           double alpha,
                           bool failforce);
                         
void Surv_Uni_Split_Cont_Pseudo(Uni_Split_Class& TempSplit, 
                                 uvec& obs_id,
                                 const vec& x,
                                 const uvec& Y, // Y is collapsed
                                 const uvec& Censor, // Censor is collapsed
                                 size_t NFail,
                                 vec& z_eta,
                                 int split_gen,
                                 int nsplit,
                                 size_t nmin, 
                                 double alpha,
                                 bool failforce);
   
 
 
void Surv_Uni_Split_Cat(Uni_Split_Class& TempSplit, 
                        uvec& obs_id,
                        const vec& x,
                        const uvec& Y, // Y is collapsed
                        const uvec& Censor, // Censor is collapsed
                        size_t NFail,
                        const uvec& All_Fail,
                        const vec& All_Risk,
                        vec& Temp_Vec,
                        double penalty,
                        int split_gen,
                        int split_rule,
                        int nsplit,
                        size_t nmin,
                        double alpha,
                        bool failforce,
                        size_t ncat);

void Surv_Uni_Split_Cat_W(Uni_Split_Class& TempSplit, 
                          uvec& obs_id,
                          const vec& x,
                          const uvec& Y, // Y is collapsed
                          const uvec& Censor, // Censor is collapsed
                          vec& obs_weight,
                          size_t NFail,
                          double penalty,
                          int split_gen,
                          int split_rule,
                          int nsplit,
                          size_t nmin, 
                          double alpha,
                          bool failforce,
                          size_t ncat);
                        
void collapse(const uvec& Y, 
              const uvec& Censor, 
              uvec& Y_collapse, 
              uvec& Censor_collapse, 
              uvec& obs_id,
              size_t& NFail);

// splitting score calculations

double logrank(const uvec& Left_Fail, 
               const uvec& Left_Risk, 
               const uvec& All_Fail, 
               const vec& All_Risk);

double suplogrank(const uvec& Left_Fail, 
                  const uvec& Left_Risk, 
                  const uvec& All_Fail, 
                  const vec& All_Risk,
                  vec& Temp_Vec);

double CoxGrad(uvec& Pseudo_X,
           const vec& z_eta);

// prediction 

void Surv_Uni_Forest_Pred(cube& Pred,
                          const Surv_Uni_Forest_Class& SURV_FOREST,
                          const mat& X,
                          const uvec& Ncat,
                          size_t NFail,
                          const uvec& treeindex,
                          int usecores,
                          int verbose);

#endif

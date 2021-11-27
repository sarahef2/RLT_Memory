// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// RegUniForestFit
List RegUniForestFit(arma::mat& X, arma::vec& Y, arma::uvec& Ncat, arma::vec& obsweight, arma::vec& varweight, arma::umat& ObsTrack, List& param);
RcppExport SEXP _RLT_RegUniForestFit(SEXP XSEXP, SEXP YSEXP, SEXP NcatSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP paramSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obsweight(obsweightSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type varweight(varweightSEXP);
    Rcpp::traits::input_parameter< arma::umat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< List& >::type param(paramSEXP);
    rcpp_result_gen = Rcpp::wrap(RegUniForestFit(X, Y, Ncat, obsweight, varweight, ObsTrack, param));
    return rcpp_result_gen;
END_RCPP
}
// RegUniForestPred
List RegUniForestPred(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeAve, arma::mat& X, arma::uvec& Ncat, arma::uvec& treeindex, bool keep_all, int usecores, int verbose);
RcppExport SEXP _RLT_RegUniForestPred(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeAveSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP treeindexSEXP, SEXP keep_allSEXP, SEXP usecoresSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeAve(NodeAveSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type treeindex(treeindexSEXP);
    Rcpp::traits::input_parameter< bool >::type keep_all(keep_allSEXP);
    Rcpp::traits::input_parameter< int >::type usecores(usecoresSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(RegUniForestPred(SplitVar, SplitValue, LeftNode, RightNode, NodeAve, X, Ncat, treeindex, keep_all, usecores, verbose));
    return rcpp_result_gen;
END_RCPP
}
// EofVar
List EofVar(arma::mat& Pred, int usecores, int verbose);
RcppExport SEXP _RLT_EofVar(SEXP PredSEXP, SEXP usecoresSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type Pred(PredSEXP);
    Rcpp::traits::input_parameter< int >::type usecores(usecoresSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(EofVar(Pred, usecores, verbose));
    return rcpp_result_gen;
END_RCPP
}
// ARMA_EMPTY_UMAT
arma::umat ARMA_EMPTY_UMAT();
RcppExport SEXP _RLT_ARMA_EMPTY_UMAT() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(ARMA_EMPTY_UMAT());
    return rcpp_result_gen;
END_RCPP
}
// ARMA_EMPTY_VEC
arma::vec ARMA_EMPTY_VEC();
RcppExport SEXP _RLT_ARMA_EMPTY_VEC() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(ARMA_EMPTY_VEC());
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RLT_RegUniForestFit", (DL_FUNC) &_RLT_RegUniForestFit, 7},
    {"_RLT_RegUniForestPred", (DL_FUNC) &_RLT_RegUniForestPred, 11},
    {"_RLT_EofVar", (DL_FUNC) &_RLT_EofVar, 3},
    {"_RLT_ARMA_EMPTY_UMAT", (DL_FUNC) &_RLT_ARMA_EMPTY_UMAT, 0},
    {"_RLT_ARMA_EMPTY_VEC", (DL_FUNC) &_RLT_ARMA_EMPTY_VEC, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_RLT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

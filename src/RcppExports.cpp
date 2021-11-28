// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

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
// UniKernel_Self
List UniKernel_Self(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::mat& X, arma::uvec& Ncat, size_t verbose);
RcppExport SEXP _RLT_UniKernel_Self(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(UniKernel_Self(SplitVar, SplitValue, LeftNode, RightNode, X, Ncat, verbose));
    return rcpp_result_gen;
END_RCPP
}
// UniKernel_Cross
List UniKernel_Cross(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::mat& X1, arma::mat& X2, arma::uvec& Ncat, size_t verbose);
RcppExport SEXP _RLT_UniKernel_Cross(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP X1SEXP, SEXP X2SEXP, SEXP NcatSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X1(X1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X2(X2SEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(UniKernel_Cross(SplitVar, SplitValue, LeftNode, RightNode, X1, X2, Ncat, verbose));
    return rcpp_result_gen;
END_RCPP
}
// UniKernel_Train
List UniKernel_Train(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::mat& X1, arma::mat& X2, arma::uvec& Ncat, arma::umat& ObsTrack, size_t verbose);
RcppExport SEXP _RLT_UniKernel_Train(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP X1SEXP, SEXP X2SEXP, SEXP NcatSEXP, SEXP ObsTrackSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X1(X1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X2(X2SEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::umat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(UniKernel_Train(SplitVar, SplitValue, LeftNode, RightNode, X1, X2, Ncat, ObsTrack, verbose));
    return rcpp_result_gen;
END_RCPP
}
// RegMultiForestFit
List RegMultiForestFit(arma::mat& X, arma::vec& Y, arma::uvec& Ncat, arma::vec& obsweight, arma::vec& varweight, arma::umat& ObsTrack, List& param);
RcppExport SEXP _RLT_RegMultiForestFit(SEXP XSEXP, SEXP YSEXP, SEXP NcatSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP paramSEXP) {
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
    rcpp_result_gen = Rcpp::wrap(RegMultiForestFit(X, Y, Ncat, obsweight, varweight, ObsTrack, param));
    return rcpp_result_gen;
END_RCPP
}
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
List RegUniForestPred(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeAve, arma::mat& X, arma::uvec& Ncat, bool VarEst, bool keep_all, size_t usecores, size_t verbose);
RcppExport SEXP _RLT_RegUniForestPred(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeAveSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP VarEstSEXP, SEXP keep_allSEXP, SEXP usecoresSEXP, SEXP verboseSEXP) {
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
    Rcpp::traits::input_parameter< bool >::type VarEst(VarEstSEXP);
    Rcpp::traits::input_parameter< bool >::type keep_all(keep_allSEXP);
    Rcpp::traits::input_parameter< size_t >::type usecores(usecoresSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(RegUniForestPred(SplitVar, SplitValue, LeftNode, RightNode, NodeAve, X, Ncat, VarEst, keep_all, usecores, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RLT_ARMA_EMPTY_UMAT", (DL_FUNC) &_RLT_ARMA_EMPTY_UMAT, 0},
    {"_RLT_ARMA_EMPTY_VEC", (DL_FUNC) &_RLT_ARMA_EMPTY_VEC, 0},
    {"_RLT_UniKernel_Self", (DL_FUNC) &_RLT_UniKernel_Self, 7},
    {"_RLT_UniKernel_Cross", (DL_FUNC) &_RLT_UniKernel_Cross, 8},
    {"_RLT_UniKernel_Train", (DL_FUNC) &_RLT_UniKernel_Train, 9},
    {"_RLT_RegMultiForestFit", (DL_FUNC) &_RLT_RegMultiForestFit, 7},
    {"_RLT_RegUniForestFit", (DL_FUNC) &_RLT_RegUniForestFit, 7},
    {"_RLT_RegUniForestPred", (DL_FUNC) &_RLT_RegUniForestPred, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_RLT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

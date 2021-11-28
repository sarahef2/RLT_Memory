//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Random Forest Kernel
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Tree_Functions.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List UniKernel_Self(arma::field<arma::ivec>& SplitVar,
                    arma::field<arma::vec>& SplitValue,
                    arma::field<arma::uvec>& LeftNode,
                    arma::field<arma::uvec>& RightNode,
                    arma::mat& X,
                    arma::uvec& Ncat,
                    size_t ncores,
                    size_t verbose)
{
  size_t N = X.n_rows;
  size_t ntrees = SplitVar.n_elem; 
  
  // check number of cores
  size_t usecores = checkCores(ncores, verbose);
  
  // initiate output kernel
  // each element for one testing subject 
  
  arma::ucube Kernel(N, N, usecores, fill::zeros);
  uvec real_id = linspace<uvec>(0, N-1, N);  
  
#pragma omp parallel num_threads(usecores)
{
  #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      DEBUG_Rcout << "--- on tree " << nt << std::endl;
      
      Uni_Tree_Class OneTree(SplitVar(nt),
                             SplitValue(nt),
                             LeftNode(nt),
                             RightNode(nt));
      
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      // get terminal node id
      Uni_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      // record
      size_t tid = omp_get_thread_num();      
      
      uvec UniqueNode = unique(TermNode);
      
      for (auto j : UniqueNode)
      {
        uvec ID = real_id(find(TermNode == j));
        
        Kernel.slice(tid).submat(ID, ID) += 1;
      }
    }
}

    umat K(N, N, fill::zeros);
    
    for (size_t j = 0; j < usecores; j++)
    {
      K += Kernel.slice(j);
    }
    
    List ReturnList;
    ReturnList["Kernel"] = K;
    
    return(ReturnList);
}

// [[Rcpp::export()]]
List UniKernel_Cross(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& X2,
                     arma::uvec& Ncat,
                     size_t ncores,
                     size_t verbose)
{
  Rcout << "/// RLT Kernel Function Cross is not avaliable yet ///" << std::endl;
  
  List ReturnList;
  ReturnList["Kernel"] = 0;
  
  return(ReturnList);
  
}

// [[Rcpp::export()]]
List UniKernel_Train(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& XTrain,
                     arma::uvec& Ncat,
                     arma::umat& ObsTrack,
                     size_t ncores,
                     size_t verbose)
{
  Rcout << "/// RLT Kernel Function vs.train is not avaliable yet ///" << std::endl;
  
  List ReturnList;
  ReturnList["Kernel"] = 0;
  
  return(ReturnList);
  
}
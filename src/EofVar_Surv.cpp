//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Estimate the Expectation of Variance
//  **********************************

// Adding headers
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility/Utility.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List EofVar_S(arma::umat& ObsTrack,
            arma::cube& Pred,
            arma::cube& Pred_TV,
            arma::uvec& C,
            int usecores,
            int verbose)
{
  DEBUG_Rcout << "-- calculate E(Var(Tree|C Shared)) ---" << std::endl;
  
  DEBUG_Rcout << C << std::endl;
  
  usecores = checkCores(usecores, verbose);
  
  size_t N = Pred.n_slices;
  size_t ntrees = Pred.n_cols;
  size_t length = C.n_elem;
  size_t tmpts = Pred.n_rows;
  size_t ntree_half = ntrees/2;
  
  // arma::cube surv(tmpts, ntrees, N);
  // arma::cube surv_tree(tmpts, ntrees*10, N);
  // for(size_t n=0; n < N; n++){
  //   surv.slice(n) = exp(-cumsum(Pred.slice(n), 0));
  //   surv_tree.slice(n) = exp(-cumsum(Pred.slice(n), 0));
  // }
  
  arma::mat Tree_Var_Est(N, tmpts, fill::zeros);
  arma::mat tmp_slice;
  arma::mat tmp_diff;
  arma::cube Tree_Cov_Est(tmpts, tmpts, N, fill::zeros);
//#pragma omp parallel num_threads(usecores)
//{
//#pragma omp for schedule(dynamic)
  for(size_t n = 0; n < N; n++){
    tmp_slice = Pred.slice(n);//surv_tree.slice(n);//Pred_TV.slice(n);
    tmp_diff = tmp_slice.cols(0,(ntree_half-1)) - 
      tmp_slice.cols(ntree_half,(ntree_half*2-1));
    for(size_t nt = 0; nt < ntree_half; nt++){
      for(size_t tm = 0; tm < tmpts; tm++){
        // Tree_Var_Est(n, tm) += (tmp(tm, nt) - tmp(tm, nt+ntree_half)) * 
        //   (tmp(tm, nt) - tmp(tm, nt+ntree_half))/2;
        // Tree_Cov_Est(tm, tm, n) += (tmp(tm, nt) - tmp(tm, nt+ntree_half)) * 
        //   (tmp(tm, nt) - tmp(tm, nt+ntree_half))/2;
        for(size_t tm2 = tm; tm2 < tmpts; tm2++){
          //Tree_Cov_Est(tm, tm2, n) = sum(tmp_diff.row(tm) * 
           // tmp_diff.row(tm2)/2);
          Tree_Cov_Est(tm, tm2, n) += (tmp_slice(tm, nt) - tmp_slice(tm, nt+ntree_half)) * 
            (tmp_slice(tm2, nt) - tmp_slice(tm2, nt+ntree_half))/2;
          if(tm==tm2){
            //Tree_Var_Est(n, tm) = sum(tmp_diff.row(tm) * 
            //  tmp_diff.row(tm)/2);
            Tree_Var_Est(n, tm) += (tmp_slice(tm, nt) - tmp_slice(tm, nt+ntree_half)) *
              (tmp_slice(tm, nt) - tmp_slice(tm, nt+ntree_half))/2;
          }else{
            //Tree_Cov_Est(tm2, tm, n) = sum(tmp_diff.row(tm2) * 
            //  tmp_diff.row(tm)/2);
            Tree_Cov_Est(tm2, tm, n) += (tmp_slice(tm, nt) - tmp_slice(tm, nt+ntree_half)) * 
               (tmp_slice(tm2, nt) - tmp_slice(tm2, nt+ntree_half))/2;
          }
        }
      }
    }
      //Tree_Var_Est(n, m) = var(tmp.row(m));
  }
//}
    //Tree_Cov_Est.slice(n) = cov(tmp.t());
  Tree_Var_Est/=ntree_half;
  Tree_Cov_Est/=ntree_half;
  
  Rcout << "-- Initialize ---" << std::endl;
  //arma::cube Est(length, tmpts, N, fill::zeros);
  arma::mat Est(tmpts, N, fill::zeros);
  //arma::uvec allcounts(length, fill::zeros);
  //arma::umat pairs(ntrees*ntrees, 4, fill::zeros);

  //Rcout << "Count pairs up to "<<ntrees*ntrees << std::endl;
  //size_t rnum = 0;
  //Rcout << "Start rnum: "<<rnum << std::endl;
//#pragma omp parallel num_threads(usecores)
//{
//#pragma omp for schedule(dynamic)
  //for (size_t i = 0; i < (ntrees - 1); i++){
  //  for (size_t j = i+1; j < ntrees; j++){
  //    uvec pair = {i, j};
  //    int  incommon = sum( min(ObsTrack.cols(pair), 1) );
      //uvec tmppair = {i, j, incommon};
      //Rcout << "Incommon count: "<< incommon << std::endl;
      //Rcout << "Row num: "<<i*(ntrees-1)+(j-1-i) << std::endl;
  //    pairs(i*(ntrees-1)+(j-1-i), 0) = i;
  //    pairs(i*(ntrees-1)+(j-1-i), 1) = j;
  //    pairs(i*(ntrees-1)+(j-1-i), 2) = incommon-C(0);
      //pairs(i*(ntrees-1)+(j-1-i), 3) = incommon-C(0);
  //    if ((incommon-C(0))<length){
        //Rcout << "Index: "<<incommon - C(0) << std::endl;
  //      pairs(i*(ntrees-1)+(j-1-i), 3) = 1;
  //      allcounts(incommon - C(0))++;
  //    }
      //rnum++;
      //Rcout << rnum << std::endl;
      //if ( C(0)<=incommon and C(length-1)>=incommon){
      //    allcounts(incommon - length)++;
      //    pairs.insert_rows(0, tmppair);
      //  }
      //}
  //  }
//}
  
  //arma::uvec wi_C = find(pairs.col(3)>=C(0) and pairs.col(3)<=C(length-1));
  //arma::umat pairs_red = pairs.rows(wi_C);
  
  //size_t pair_count = pairs.n_rows;
  //Rcout << "Pair counts "<< pair_count << std::endl;
  //Rcout << "pairs "<< pairs.rows(0,4) << std::endl;
  //arma::mat wcsigma(N, tmpts);
  arma::cube wcsigmaCov(tmpts, tmpts, N);
  arma::mat temp2;
  
  //Rcout << "Starting paralell computing" << std::endl;
//#pragma omp parallel num_threads(usecores)
  //{
  //#pragma omp for schedule(dynamic)
  //   for(size_t n = 0; n < N; n++){
  //     //Rcout << "n: "<<n << std::endl;
  //     arma::cube Cov_Est(tmpts, tmpts, length, fill::zeros);
  //     for(size_t k = 0; k < pair_count; k++){
  //       //Rcout << "k: "<<k << std::endl;
  //       //Rcout << "pairs(k, 2): "<<pairs(k, 2) << " pairs(k, 3): "<<pairs(k, 3) << std::endl;
  //       if(pairs(k, 3)==1){
  //         //Rcout << "k: "<<k << std::endl;
  //         for(size_t tm = 0; tm < tmpts; tm++){
  //            Est(pairs(k, 2), tm, n) += 0.5 * (Pred(tm,pairs(k, 0),n) - Pred(tm,pairs(k, 1),n)) *
  //              (Pred(tm,pairs(k, 0),n) - Pred(tm,pairs(k, 1),n))/allcounts(pairs(k,2));
  //           for(size_t tm2 = 0; tm2 < tmpts; tm2++){
  //             Cov_Est(tm, tm2, pairs(k, 2)) += 0.5 * (Pred(tm,pairs(k, 0),n) - Pred(tm2,pairs(k, 1),n)) *
  //               (Pred(tm,pairs(k, 0),n) - Pred(tm2,pairs(k, 1),n))/allcounts(pairs(k,2));
  //             }
  //           }
  //       }
  //     }
  //     //Rcout << "Cov_Est "<< Cov_Est.slice(0) << std::endl;
  //     arma::mat test = sum(Cov_Est, 2);
  //     //Rcout << "Sum over slices "<< test << std::endl;
  //     //Rcout << "Dim of Sum over slices "<< test.n_rows <<" "<<test.n_cols << std::endl;
  //     //Rcout << "Dim of slice "<< wcsigmaCov.slice(n).n_rows <<" "<<wcsigmaCov.slice(n).n_cols << std::endl;
  //     wcsigmaCov.slice(n) = sum(Cov_Est, 2);
  //   }
  // //}
  arma::cube Cov_Est(tmpts, tmpts, N, fill::zeros);
//#pragma omp parallel num_threads(usecores)
//{
//#pragma omp for schedule(dynamic)
  for(size_t n = 0; n < N; n++){
    size_t count = 0;
    //Rcout << "n: "<<n << std::endl;
  for (size_t i = 0; i < (ntree_half - 1); i++){
      for (size_t j = i+1; j < ntree_half/2; j++){
        count++;
        for(size_t tm = 0; tm < tmpts; tm++){
          for(size_t tm2 = tm; tm2 < tmpts; tm2++){
            //Cov_Est(tm, tm2, n) += 0.5 * (surv(tm,i,n) - surv(tm2,j,n)) *
            //  (surv(tm,i,n) - surv(tm2,j,n));
            Cov_Est(tm, tm2, n) += 0.5 * (Pred(tm,i,n) - Pred(tm,j,n)) *
              (Pred(tm2,i,n) - Pred(tm2,j,n));
            if(tm==tm2){
              Est(tm, n) += 0.5 * (Pred(tm,i,n) - Pred(tm,j,n)) *
                (Pred(tm,i,n) - Pred(tm,j,n));
              //Est(tm, n) += 0.5 * (surv(tm,i,n) - surv(tm,j,n)) *
              //  (surv(tm,i,n) - surv(tm,j,n));
            }else{
              Cov_Est(tm2, tm, n) += 0.5 * (Pred(tm,i,n) - Pred(tm,j,n)) *
                (Pred(tm2,i,n) - Pred(tm2,j,n));
              //Cov_Est(tm2, tm, n) += 0.5 * (surv(tm2,i,n) - surv(tm,j,n)) *
              //  (surv(tm2,i,n) - surv(tm,j,n));
            }
          }
        }
        
      }
    }
//}
    Cov_Est.slice(n)/=(count);//*count
    Est.col(n)/=(count);//*count
  }
//}
  
  
  // #pragma omp parallel num_threads(usecores)
  // {
  // #pragma omp for schedule(dynamic)
  //       for (size_t l = 0; l < length; l++){ // calculate all C values
  //         arma::cube Cov_Est(tmpts, tmpts, N, fill::zeros);
  //         size_t count = 0;
  //     
  //     for (size_t i = 0; i < (ntrees - 1); i++){
  //       for (size_t j = i+1; j < ntrees; j++){
  //         
  //         uvec pair = {i, j};
  //         pairs.insert_rows(0, {i, j, C(l)});
  //         
  //         if ( sum( min(ObsTrack.cols(pair), 1) ) == C(l) )
  //         {
  //           count++;
  //           for(size_t n = 0; n < N; n++){
  //             for(size_t tm = 0; tm < tmpts; tm++){
  //                 Est(l, tm, n) += 0.5 * (Pred(tm,i,n) - Pred(tm,j,n)) *
  //                 (Pred(tm,i,n) - Pred(tm,j,n));
  //               for(size_t tm2 = 0; tm2 < tmpts; tm2++){
  //                 Cov_Est(tm, tm2, n) += 0.5 * (Pred(tm,i,n) - Pred(tm2,j,n)) *
  //                   (Pred(tm,i,n) - Pred(tm2,j,n));
  //               }
  //             }
  //           }
  //         }
  //       }}
  //     
  //     Est.row(l) /= count;
  //     allcounts(l) = count;
  //     Cov_Est /= count;
  //   }
  // }

  //DEBUG_Rcout << "-- total count  ---" << allcounts << std::endl;  
  DEBUG_Rcout << "-- all estimates  ---" << Est << std::endl; 
  
  // For a given test ob
  // Est.slice(n) length*tmpt
  // allcounts length
  // Want vector of length tmpt
  // arma::mat wcsigma(N, tmpts);
  // arma::mat temp2;

  // for(size_t n = 0; n < N; n++){
  //   temp2 = Est.slice(n);
  //   for(size_t tm = 0; tm < tmpts; tm++){
  //     wcsigma(n,tm) = sum(temp2.col(tm));// * temp2.col(tm));
  //   }
  // }
  
  //Rcout << "Matrix manipulation" << std::endl;
  //arma::mat wcsigma = sum(Est, 0);
  
  arma::mat var = Tree_Var_Est - Est;
  arma::cube cov = Tree_Cov_Est - Cov_Est;

  List ReturnList;
  
  //ReturnList["allcounts"] = allcounts;
  ReturnList["estimation"] = Est;
  ReturnList["cov.estimation"] = Cov_Est;
  ReturnList["tree.var"] = Tree_Var_Est;
  //ReturnList["wcsigma"] = wcsigma;
  //ReturnList["wcsigmaCov"] = wcsigmaCov;
  ReturnList["var"] = var;
  ReturnList["tree.cov"] = Tree_Cov_Est;
  ReturnList["cov"] = cov;
  
  return(ReturnList);
}






// List EofVar_Surv(//arma::cube& Pred,
//             //arma::cube& Pred_TV,
//             arma::umat& ObsTrack,
//             arma::mat& Pred,
//             arma::uvec& C,
//             int usecores,
//             int verbose)
// {
//   DEBUG_Rcout << "-- calculate E(Var(Tree|C Shared)) ---" << std::endl;
//   
//   DEBUG_Rcout << C << std::endl;
//   
//   usecores = checkCores(usecores, verbose);
//   
//   size_t N = Pred.n_rows;
//   size_t ntrees = Pred.n_cols;
//   //The number of C's with which we will estimate the variance
//   size_t length = C.n_elem;
// 
//   //arma::mat Tree_Var_Est(N, Pred_TV.n_rows, fill::zeros);
//   //arma::mat tmp(N, Pred_TV.n_rows, fill::zeros);
//   //for(size_t n = 0; n < N; n++){
//   //  tmp = Pred_TV.slice(n);
//   //  for(size_t m = 0; m < Pred_TV.n_rows; m++){
//   //    Tree_Var_Est(n, m) = var(tmp.row(m));
//   //  }
//   //}
//   
//   //For each observation, record the variance at each C  
//   arma::mat Est(N, length, fill::zeros);
//   //Keep track of the number of tree pairs used to calculate each C
//   arma::uvec allcounts(length, fill::zeros);
//    
// #pragma omp parallel num_threads(usecores)
// {
//   #pragma omp for schedule(dynamic)
//   for (size_t l = 0; l < length; l++) // calculate all C values
//   {
//     size_t count = 0;
//     
//     //For each pair of trees...
//     for (size_t i = 0; i < (ntrees - 1); i++){
//     for (size_t j = i+1; j < ntrees; j++){
//       
//       //Indices of the pair
//       uvec pair = {i, j};
//         
//         //Pulls the columns related to the indices
//         //Finds the minimum in each row
//         //If the minimum is 1, then that observation was included in both rows
//         //Count the number of obs used in both trees
//         //If the sum of shared obs equals C(l)...
//       if ( sum( min(ObsTrack.cols(pair), 1) ) == C(l) )
//       {
//         count++;
//         
//         //Calculate ..sigma_c and add it to the others
//         Est.col(l) += 0.5 * square(Pred.col(i) - Pred.col(j));
//       }
//     }}
//     
//     //Take the mean of ..sigma_c
//     Est.col(l) /= count;
//     //Keep the count of ..sigma_c's
//     allcounts(l) = count;
//   }
//   //We have now estimated \binom{n}{k}^{-2}sum(sum(..sigma_c))
// }
// 
//   DEBUG_Rcout << "-- total count  ---" << allcounts << std::endl;  
//   DEBUG_Rcout << "-- all estimates  ---" << Est << std::endl; 
// 
//   List ReturnList;
//   
//   ReturnList["allcounts"] = allcounts;
//   ReturnList["estimation"] = Est;
//   //ReturnList["tree.var"] = Tree_Var_Est;
//   
//   return(ReturnList);

//}









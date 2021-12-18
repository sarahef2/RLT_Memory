//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Utility Functions and Definitions
//  **********************************

// my header file
# include "../RLT.h"
# include <RcppArmadillo.h>
# include <Rcpp.h>
# include <xoshiro.h>
# include <dqrng_distribution.h>
# include <limits>

using namespace Rcpp;
using namespace arma;

// ******* //
//  Debug  //
// ******* //

// this debug function will output results to a .txt file
void printLog(const char*, const char*, const int, const double);

#ifdef RLT_DEBUG
#define DEBUGPRINT(mode, x, n1, n2) printLog(mode, x, n1, n2)
#else
#define DEBUGPRINT(mode, x, n1, n2)
#endif

// this debug function will output to R
#ifdef RLT_DEBUG
#define DEBUG_Rcout Rcout
#else
#define DEBUG_Rcout 0 && Rcout
#endif

#ifndef RLT_UTILITY
#define RLT_UTILITY

// ****************//
// Check functions //
// ****************//

size_t checkCores(size_t, size_t);

// *************//
// Calculations //
// *************//

template <class T> const T& max (const T& a, const T& b);
template <class T> const T& min (const T& a, const T& b);

// ************************//
// Random Number Generator //
// ************************//

// int intRand(const int & min, const int & max);

// Structure for Random Number generating
class Rand{
  
public:
  
  size_t seed = 0;
  dqrng::xoshiro256plus lrng; // Random Number Generator
  
  // Initialize
  Rand(size_t seed){
    
    dqrng::xoshiro256plus rng(seed);
    lrng = rng;
    
  }
  
  template<typename V>
  V random_suffle(V z){
    
   uvec temp = this -> sample(z.n_elem, 0, z.n_elem -1);
  
   V z_shuffle = z(temp); 
    
    return z_shuffle;
    
   }
  
  // Random
  size_t rand_sizet(size_t min, size_t max){
    
    boost::random::uniform_int_distribution<int> rand(min, max);
    
    return  rand(this -> lrng);
    
  };
  
  // Discrete Uniform
  arma::uvec rand_uvec(size_t Num, size_t min, size_t max){
    
    boost::random::uniform_int_distribution<int> rand(min, max);
    
    arma::uvec x(Num);
    
    for(size_t i = 0; i < Num; i++){
      
      x(i) = rand(this -> lrng);
      
    }
    
    return x;
    
  };
  
  // Uniform Distribution
  arma::vec rand_vec(size_t Num, double min, double max){

    boost::random::uniform_real_distribution<double> rand(min, max);
    
    arma::vec x(Num);
    
    for(size_t i = 0; i < Num; i++){
      
      x(i) = rand(this -> lrng);
      
    }
    
    return x;
    
  };
  
  // Sampling in a range without replacement
  arma::uvec sample(size_t Num, size_t min, size_t max) {

  
    if (max < min) max = min;

    size_t N = max - min + 1;

    arma::uvec vector = arma::linspace<uvec>(min, max, N);
    
    if (Num > N) Num = N;

    for (size_t i = 0; i < Num; i++){
      
      boost::random::uniform_int_distribution<int> rand(i, N - 1);
      
      size_t randomloc = rand(this -> lrng);

      // swap
      size_t temp = vector(i);
      vector(i) = vector(randomloc);
      vector(randomloc) = temp;
      
    }
    
    vector.resize(Num);
    
    return vector;
    
  };
  
};

#endif
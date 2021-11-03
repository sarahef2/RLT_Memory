//  **********************************
//  Reinforcement Learning Trees (RLT)
//  **********************************

// my header file
# include "../RLT.h"
# include "../Utility/Utility.h"
# include "../Trees/Trees.h"
# include "Surv_Definition.h"

using namespace Rcpp;
using namespace arma;

//Functions moved from Miscellaneous.cpp
void move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Surv_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin)
{
  // in this case, we will not be able to control for nmin
  // but extreamly small nmin should not have a high splitting score
  lowindex = 0;
  highindex = true_cat - 2;
  
  if (true_cat == 2) //nothing we can do
    return; 
  
  DEBUG_Rcout << "        --- start moving index with lowindex " << lowindex << " highindex " << highindex << std::endl;
  
  lowindex = 0;
  highindex = true_cat-2;
  size_t lowcount = cat_reduced[0].count;
  size_t highcount = cat_reduced[true_cat-1].count;
  
  // now both low and high index are not tied with the end
  if ( lowcount >= nmin and highcount >= nmin ) // everything is good
    return;
  
  if ( lowcount < nmin and highcount >= nmin ) // only need to fix lowindex
  {
    while( lowcount < nmin and lowindex <= highindex ){
      lowindex++;
      lowcount += cat_reduced[lowindex].count;
    }
    
    if ( lowindex > highindex ) lowindex = highindex;
    
    return;
    DEBUG_Rcout << "        --- case 1 with lowindex " << lowindex << " highindex " << highindex << std::endl;
  }
  
  if ( lowcount >= nmin and highcount < nmin ) // only need to fix highindex
  {
    while( highcount < nmin and lowindex <= highindex ){
      DEBUG_Rcout << "        --- adding " << cat_reduced[highindex].count << " count to highcount " << highcount << std::endl;
      highcount += cat_reduced[highindex].count;
      highindex--;
    }
    
    if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex; // sometimes highindex will be negative and turned into very large number 
    
    DEBUG_Rcout << "        --- case 2 with lowindex " << lowindex << " highindex " << highindex << std::endl;    
    
    return;
  }
  
  if ( lowcount < nmin and highcount < nmin ) // if both need to be fixed, start with one randomly
  {
    if ( intRand(0, 1) )
    { // fix lowindex first
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex].count;
      }
      
      if (lowindex > highindex ) lowindex = highindex;
      
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex].count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      DEBUG_Rcout << "        --- case 3 with lowindex " << lowindex << " highindex " << highindex << std::endl;
      return;
      
    }else{ // fix highindex first
      while( highcount < nmin and lowindex <= highindex ){
        highcount += cat_reduced[highindex].count;
        highindex--;
      }
      
      if (highindex < lowindex or highindex > true_cat - 2 ) highindex = lowindex;
      
      while( lowcount < nmin and lowindex <= highindex ){
        lowindex++;
        lowcount += cat_reduced[lowindex].count;
      }
      
      if (lowindex > highindex) lowindex = highindex;
      
      DEBUG_Rcout << "        --- case 4 with lowindex " << lowindex << " highindex " << highindex << std::endl;
      return;
    }
  }
}



double record_cat_split(size_t cat, 
                        std::vector<Surv_Cat_Class>& cat_reduced)
{
  size_t ncat = cat_reduced.size() - 1;
  uvec goright(ncat + 1, fill::zeros);
  
  for (size_t i = 0; i <= cat; i++)
    goright[cat_reduced[i].cat] = 1;
  
  return pack(ncat + 1, goright);
}

double record_cat_split(arma::uvec& goright_temp, 
                        std::vector<Surv_Cat_Class>& cat_reduced)
{
  size_t ncat = cat_reduced.size() - 1;
  uvec goright(ncat + 1, fill::zeros);
  
  for (size_t i = 0; i < goright_temp.n_elem; i++)
  {
    if (goright_temp(i) == 1)
      goright[cat_reduced[i].cat] = 1;
  }
  
  return pack(ncat + 1, goright);
}

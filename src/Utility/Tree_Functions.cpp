//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Tree arranging functions
//  **********************************

// my header file
# include "../RLT.h"
# include "Utility.h"
# include "Tree_Functions.h"

using namespace Rcpp;
using namespace arma;

// categorical variable pack

double pack(const size_t nBits, const uvec& bits) // from Andy's rf package
{
  double value = bits(nBits - 1);

  for (int i = nBits - 2; i >= 0; i--)
    value = 2.0*value + bits(i);
  
  return(value);
}

void unpack(const double pack, const size_t nBits, uvec& bits) // from Andy's rf package
{
  double x = pack;
  for (size_t i = 0; i < nBits; ++i)
  {
    bits(i) = ((size_t) x & 1) ? 1 : 0;
    x /= 2;
  }
}

bool unpack_goright(double pack, const size_t cat)
{
  for (size_t i = 0; i < cat; i++) pack /= 2;
  return(((size_t) pack & 1) ? 1 : 0);
}


// for tree building 

// get inbag and oobag samples

void oob_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const size_t size,
                 const bool replacement)
{

  inbagObs.set_size(size);
  size_t N = subj_id.size();
  
  arma::uvec oob_indicate(N); // this one will contain negative values
  oob_indicate.fill(1); // this one will contain negative values
  
  arma::uvec loc;
  
  if (replacement)
  {
    // sample the locations of id with random uniform location
    loc = randi<uvec>(size, distr_param(0, N-1));
  }else{
    // permutation
    loc = arma::randperm(N, size);
  }
  
  // inbag take those locations
  // oobag remove
  for (size_t i = 0; i < size; i++)
  {
    inbagObs[i] = subj_id[loc[i]];
    oob_indicate[loc[i]] = 0;
  }
  
  oobagObs = subj_id.elem( find(oob_indicate > 0.5) );
}


void set_obstrack(arma::umat& ObsTrack,
                  const size_t nt,
                  const size_t size,
                  const bool replacement,
                  Rand& rngl)
{
	size_t N = ObsTrack.n_rows;
	
	if (replacement)
	{
	  ObsTrack.col(nt) = rngl.rand_uvec(size, 0, N-1);
		
	}else{
		
		ObsTrack.col(nt) = rngl.sample(size, 0, N-1);
	  
	}
}


// get inbag and oobag samples from ObsTrackPre

void get_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const arma::uvec& ObsTrack_nt)
{
	size_t N = sum(ObsTrack_nt);
	
	oobagObs = subj_id.elem( find(ObsTrack_nt == 0) );
	
	inbagObs.set_size(N);
	
	size_t mover = 0;
	
	for (size_t i = 0; i < ObsTrack_nt.n_elem; i++)
	{
		if (ObsTrack_nt(i) > 0)
		{
			for (size_t k = 0; k < ObsTrack_nt(i); k++)
			{
				inbagObs(mover) = subj_id(i);
				mover ++;
			}
		}
	}
}






// moveindex (continuous variable) so that both low and high are not at a tie location 
// and has sufficient number of observations

void move_cont_index(size_t& lowindex, size_t& highindex, const vec& x, const uvec& indices, size_t nmin)
{
  // in this case, we will not be able to control for nmin
  // but extremely small nmin should not have a high splitting score
  
  // size_t N = indices.size();
  
  lowindex = 0;
  highindex = indices.size()-2;
  
  while( x(indices(lowindex)) == x(indices(lowindex+1)) ) lowindex++;
  while( x(indices(highindex)) == x(indices(highindex+1)) ) highindex--;
  
  // now both low and high index are not tied with the end
  if ( lowindex >= (nmin - 1) and highindex <= (indices.size() - nmin - 1) ) // everything is good
    return;
  
  if ( x(indices(lowindex)) == x(indices(highindex)) ) // nothing we can do
    return;
    
  if ( lowindex < (nmin - 1) and highindex <= (indices.size() - nmin - 1) ) // only need to fix lowindex
  {
    while( (x(indices(lowindex)) == x(indices(lowindex+1)) or lowindex < (nmin - 1)) and lowindex <= highindex ) lowindex++;
  }
  
  if ( lowindex >= (nmin - 1) and highindex > (indices.size() - nmin - 1) ) // only need to fix highindex
  {
    while( (x(indices(highindex)) == x(indices(highindex+1)) or highindex > (indices.size() - nmin - 1)) and lowindex <= highindex ) highindex--;
  }  
  
  if ( lowindex < (nmin - 1) and highindex > (indices.size() - nmin - 1) ) // if both need to be fixed, start with one randomly
  {
    if ( intRand(0, 1) )
    { // fix lowindex first
      while( (x(indices(lowindex)) == x(indices(lowindex+1)) or lowindex < (nmin - 1)) and lowindex <= highindex ) lowindex++;
      while( (x(indices(highindex)) == x(indices(highindex+1)) or highindex > (indices.size() - nmin - 1)) and lowindex <= highindex ) highindex--;
    }else{ // fix highindex first
      while( (x(indices(highindex)) == x(indices(highindex+1)) or highindex > (indices.size() - nmin - 1)) and lowindex <= highindex ) highindex--;         
      while( (x(indices(lowindex)) == x(indices(lowindex+1)) or lowindex < (nmin - 1)) and lowindex <= highindex ) lowindex++;
    }
  }
}

// construct id vectors for left and right nodes

void split_id(const vec& x, double value, uvec& left_id, uvec& obs_id) // obs_id will be treated as the right node
{
  size_t RightN = obs_id.n_elem - 1;
  size_t LeftN = 0;
  size_t i = 0;
  
  while( i <= RightN ){
    
    if ( x(obs_id(i)) <= value )
    {
      // move subject to left 
      left_id(LeftN++) = obs_id(i);
      
      // remove subject from right 
      obs_id(i) = obs_id( RightN--);
    }else{
      i++;
    }
  }
  
  left_id.resize(LeftN);
  obs_id.resize(RightN+1);
}

void split_id_cat(const vec& x, double value, uvec& left_id, uvec& obs_id, size_t ncat) // obs_id will be treated as the right node
{
  uvec goright(ncat + 1, fill::zeros); // the first (0-th) element (category) of goright will always be set to 0 --- go left, but this category does not exist.
  unpack(value, ncat + 1, goright);

  size_t RightN = obs_id.n_elem - 1;
  size_t LeftN = 0;
  size_t i = 0;
  
  while( i <= RightN ){
    
    if ( goright[x(obs_id(i))] == 0 )
    {
      // move subject to left 
      left_id(LeftN++) = obs_id(i);
      
      // remove subject from right 
      obs_id(i) = obs_id( RightN--);
    }else{
      i++;
    }
  }
  
  left_id.resize(LeftN);
  obs_id.resize(RightN+1);
}

// ****************//
// field functions //
// ****************//

void field_vec_resize(arma::field<arma::vec>& A, size_t size)
{
  arma::field<arma::vec> B(size);
  
  size_t common_size = (A.n_elem > size) ? size : A.n_elem;

  for (size_t i = 0; i < common_size; i++)
  {
    B[i] = vec(A[i].begin(), A[i].size(), false, true);
  }
  
  A.set_size(size);
  for (size_t i = 0; i < common_size; i++)
  {
    A[i] = vec(B[i].begin(), B[i].size(), false, true);
  }
}

void field_vec_resize(arma::field<arma::uvec>& A, size_t size)
{
  arma::field<arma::uvec> B(size);
  
  size_t common_size = (A.n_elem > size) ? size : A.n_elem;
  
  for (size_t i = 0; i < common_size; i++)
  {
    B[i] = uvec(A[i].begin(), A[i].size(), false, true);
  }
  
  A.set_size(size);
  for (size_t i = 0; i < common_size; i++)
  {
    A[i] = uvec(B[i].begin(), B[i].size(), false, true);
  }
}

// for categorical variables

bool cat_reduced_compare(Cat_Class& a, Cat_Class& b)
{
    if (a.count == 0 and b.count == 0)
        return 0;
    
    if (a.count > 0 and b.count == 0)
        return 1;
    
    if (a.count == 0 and b.count > 0)
        return 0;
    
    return ( a.score < b.score );
}

bool cat_reduced_collapse(Cat_Class& a, Cat_Class& b)
{
    if (a.count > 0 and b.count == 0)
        return 1;
  
    return 0;
}


/*
bool cat_reduced_compare_score(Cat_Class& a, Cat_Class& b)
{
    return ( a.score < b.score );
}
*/

void goright_roller(arma::uvec& goright_cat)
{
  size_t n = goright_cat.n_elem;
  
  for (size_t i = 0; i < n-1; i ++)
  {
    if (goright_cat(i) == 2)
    {
      goright_cat(i) = 0;
      goright_cat(i+1)++;
    }
  }
}

// Find the terminal node for X in one tree
void Uni_Find_Terminal_Node(size_t Node, 
                            const Uni_Tree_Class& OneTree,
                            const mat& X,
                            const uvec& Ncat,
                            uvec& proxy_id,
                            const uvec& real_id,
                            uvec& TermNode)
{
  
  size_t size = proxy_id.n_elem;
  
  //If the current node is a terminal node
  if (OneTree.SplitVar[Node] == -1)
  {
    // For all the observations in the node,
    // Set its terminal node
    for ( size_t i=0; i < size; i++ )
      TermNode(proxy_id(i)) = Node;
  }else{
    
    uvec id_goright(proxy_id.n_elem, fill::zeros);
    
    size_t SplitVar = OneTree.SplitVar(Node);
    double SplitValue = OneTree.SplitValue(Node);
    double xtemp = 0;
    
    if ( Ncat(SplitVar) > 1 ) // categorical var 
    {
      
      uvec goright(Ncat(SplitVar) + 1);
      unpack(SplitValue, Ncat(SplitVar) + 1, goright); // from Andy's rf package
      
      for (size_t i = 0; i < size ; i++)
      {
        xtemp = X( real_id( proxy_id(i) ), SplitVar);
        
        if ( goright( (size_t) xtemp ) == 1 )
          id_goright(i) = 1;
      }
      
    }else{
      
      //For the obs in the current internal node
      for (size_t i = 0; i < size ; i++)
      {
        //Determine the x values for this variable
        xtemp = X( real_id( proxy_id(i) ), SplitVar);
        
        //If they are greater than the value, go right
        if (xtemp > SplitValue)
          id_goright(i) = 1;
      }
    }
    
    //All others go left
    uvec left_proxy = proxy_id(find(id_goright == 0));
    proxy_id = proxy_id(find(id_goright == 1));
    
    // left node 
    
    if (left_proxy.n_elem > 0)
    {
      Uni_Find_Terminal_Node(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode);
    }
    
    // right node
    if (proxy_id.n_elem > 0)
    {
      Uni_Find_Terminal_Node(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode);      
    }
    
  }
  
  return;
  
}


//Function for variable importance
void Uni_Find_Terminal_Node_ShuffleJ(size_t Node, 
                                     const Uni_Tree_Class& OneTree,
                                     const mat& X,
                                     const uvec& Ncat,
                                     uvec& proxy_id,
                                     const uvec& real_id,
                                     uvec& TermNode,
                                     const vec& tildex,
                                     const size_t j)
{
  
  size_t size = proxy_id.n_elem;
  
  //If terminal node
  if (OneTree.SplitVar[Node] == -1)
  {
    for ( size_t i=0; i < size; i++ )
      TermNode(proxy_id(i)) = Node;
  }else{
    
    uvec id_goright(proxy_id.n_elem, fill::zeros);
    
    size_t SplitVar = OneTree.SplitVar(Node);
    double SplitValue = OneTree.SplitValue(Node);
    double xtemp = 0;
    
    if ( Ncat(SplitVar) > 1 ) // categorical var 
    {
      
      uvec goright(Ncat(SplitVar) + 1);
      unpack(SplitValue, Ncat(SplitVar) + 1, goright); // from Andy's rf package
      
      for (size_t i = 0; i < size ; i++)
      {
        if (SplitVar == j)
        {
          xtemp = tildex( proxy_id(i) );
          
        }else{
          xtemp = X( real_id( proxy_id(i) ), SplitVar);
        }
        
        
        
        if ( goright( (size_t) xtemp ) == 1 )
          id_goright(i) = 1;
      }
      
    }else{
      
      for (size_t i = 0; i < size ; i++)
      {
        if (SplitVar == j)
        {
          // If it is the shuffle variable, randomly get x
          xtemp = tildex( proxy_id(i) );
        }else{
          xtemp = X( real_id( proxy_id(i) ), SplitVar);
        }
        
        if (xtemp > SplitValue)
          id_goright(i) = 1;
      }
    }
    
    uvec left_proxy = proxy_id(find(id_goright == 0));
    proxy_id = proxy_id(find(id_goright == 1));
    
    // left node 
    
    if (left_proxy.n_elem > 0)
    {
      Uni_Find_Terminal_Node_ShuffleJ(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode, tildex, j);
    }
    
    // right node
    if (proxy_id.n_elem > 0)
    {
      Uni_Find_Terminal_Node_ShuffleJ(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode, tildex, j);      
    }
    
  }
  
  return;
  
}
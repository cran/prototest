#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

#include <set>
#include <vector>
using namespace std;

#define MAXIT 1000
#define TOL 0.000001
#define ALPHA 0.05
#define BETA 0.5

void print_vector (const arma::colvec& vec){
  for (int i = 0; i < vec.n_rows; i++){
    Rcout << vec[i] << " ";
  }
  Rcout << arma::endl;
}

void print_std_vector (const std::vector<int>& vec){
  for (int i = 0; i < vec.size(); i++){
    Rcout << vec[i] << " ";
  }
  Rcout << arma::endl;
}

void print_matrix (const arma::mat& m){
  for (int i = 0; i < m.n_rows; i++){
    for (int j = 0; j < m.n_cols; j++){
      Rcout << m.row(i)[j] << " ";
    }
    Rcout << arma::endl;
  }
}

/*
 * Fixs the group indices and counts the number of unique group labels
 */
std::vector<int> fix_groups (const IntegerVector& groups, int& num_groups){
  int length = groups.size();
  std::vector<int> fixed_groups; // will hold fixed groups
  std::map<int, int> group_index_map; // mapping from old group to fixed ones
  int current_group = 0;

  for (int i = 0; i < length; i++){
    int group = groups[i];
    std::map<int, int>::iterator it = group_index_map.find(group);
    if (it == group_index_map.end()){ //haven't seen this group
      group_index_map[group] = current_group;
      current_group++;
    }
    fixed_groups.push_back(group_index_map[group]);
  }

  // return
  num_groups = current_group;
  return fixed_groups;
}

/*
 * Computes the loglikelihood at the current value of theta. Stores the result in the parameter 'loglik', which is provided by the user
 * Returns true/false depending on whether the determinant is positive or negative. Useful info for the backtracking step later on.
 * Assumes that G is precomputed (it is the inverse of I - \sum_k \theta_k P_k
 */
bool loglikelihood (double& loglik, const arma::colvec& theta, const arma::colvec& y, const arma::mat& y_hat, const arma::mat& G, double sigma){
  // fail fast -- ingest the G matrix and compute the determinant
  // if sign is negative, return false
  double val, sign;
  arma::log_det(val, sign, G);
  if (sign <= 0){
    return false;
  }

  // compute the log likelihood
  arma::colvec res = y - y_hat*theta;
  loglik = 0.5*dot(res, res)/sigma/sigma + val; // remember that G is the inverse of the matrix in question
  return true;
}


/*
 * Computes the gradient of the prototype ML objective using the lasso projection matrices
 * Assumes precomputed Y_hat*y and Y_hat*Y_hat -- these are the standard gradient components
 * Assumes a precomputed inverse of the covariance matrix (premultiplied by V) (GV) and a matrix with the eigenvectors of the projection matrices in its columns (V)
 * Finally, uses a grouping vector (same length as there are columns in V) to ascertain to which projection matrix a eigenvector belongs
 * Uses the formula grad_k = \sum_j v_j G v_j
 */
arma::colvec gradient (const arma::colvec& theta, const arma::mat& GV, const arma::mat& V, const std::vector<int>& groups, const arma::colvec& y_hat_y, const arma::mat& y_hat_y_hat, double sigma){
  // data dimensions
  int cols = V.n_cols;

  // initial gradient
  arma::colvec grad = -y_hat_y/sigma/sigma + y_hat_y_hat*theta/sigma/sigma;

  // now loop over the columns of V and add the additional gradeint components
  //arma::mat GV_arma = G_arma*V_arma;
  for (int c = 0; c < cols; c++){
    grad[groups[c]] += dot (V.col(c), GV.col(c));
  }

  return grad;
}


/*
 * Computes the Hessian of the prototype ML using lasso porjection matrices
 */
arma::mat hessian (const arma::mat& GV, const arma::mat& V, const std::vector<int>& groups, const arma::mat& y_hat_y_hat, double sigma){
  // data dimensions
  int cols = V.n_cols;

  // initial Hessian computation
  arma::mat hess = y_hat_y_hat/sigma/sigma;
  
  // loop over the columns in V and update the Hessian entries as we go
  for (int c = 0; c < cols; c++){
    for (int d = c; d < cols; d++){
      int group_c = groups[c];
      int group_d = groups[d];
      double dot_prod = dot (V.col(c), GV.col(d));
      double dot_prod_sq = dot_prod*dot_prod;
      
      if (group_c != group_d){ // cross groups -- update the symmetric counterpart in the Hessian
	hess(group_c, group_d) += dot_prod_sq;
	hess(group_d, group_c) += dot_prod_sq;
      }else{ // same group -- two cases: c == d (only add one) and c != d (add two)
	hess(group_c, group_d) += dot_prod_sq;
	if (c != d){
	  hess(group_c, group_d) += dot_prod_sq;
	}
      }
    }
  }

  return hess;
}


/*
 * Function for computing the Y.hat matrix from a given theta
 * and the eigenvectors of the projection matrices
 */
arma::mat compute_y_hat (const arma::colvec& y, const arma::mat& V, const std::vector<int>& groups, int group_count){
  // data dimensions
  int rows = V.n_rows;
  int cols = V.n_cols;

  // compute the Y hat matrix
  arma::mat Yhat (rows, group_count, arma::fill::zeros);
  arma::mat Vy = trans(V)*y;

  for (int c = 0; c < cols; c++){
    Yhat.col(groups[c]) += Vy[c]*V.col(c);
  }

  return Yhat;
}



/*
 * Function for computing the inverse of I - \sum_k \theta_k P_k
 * Uses the Sherman-Morrison formula recursively to build it up.
 */
arma::mat sher_morr_inv(const arma::mat& V, const arma::colvec& theta, const std::vector<int>& groups, bool sm){
  // data dimensions
  int rows = V.n_rows;
  int cols = V.n_cols;

  if (sm){ // do we want the Sherman-Morrison computation?

    // loop over columns of V (and entires of theta)
    // compute inverse from Sherman-Morrison 
    arma::mat inv (rows, rows, arma::fill::eye);
    for (int c = 0; c < cols; c++){
      arma::colvec v = V.col(c);
      arma::colvec w = inv*v;
      int gr = groups[c];
      
      double coef = theta[gr]/(1-theta[gr]*dot(w, v));

      inv = inv + coef*w*trans(w);
    }

    return inv;
  }

  // once we get here, we want to compute the naive inverse
  arma::mat G = arma::eye(rows, rows);
  for (int c = 0; c < cols; c++){
    arma::colvec v = V.col(c);
    int gr = groups[c];
    G -= theta[gr]*v*trans(v);
  }
  return inv(G);

}


/*
 * Updates the estimate of the mu parameter for a given value of theta
 * Update is mu = (G1'Gy)/(G1'G1)
 */
double update_mu_elr(const arma::colvec& theta, const arma::colvec& y, const arma::mat& V, const arma::colvec& Vy, const arma::colvec& V1, const std::vector<int>& groups){
  int num_cols = V.n_cols;
  int n = y.n_rows;

  arma::colvec Gy = y;
  arma::colvec G1(n, arma::fill::ones);
  for (int c = 0; c < num_cols; c++){
    Gy -= theta[groups[c]]*Vy[c]*V.col(c);
    G1 -= theta[groups[c]]*V1[c]*V.col(c);
  }

  return dot(G1, Gy)/dot(G1, G1);
}


/*
 * Performs the Newton-Raphson optimisation for the response y and eigenvector matrix V (with grouping 'groups') and error variance sigma^2
 * Starts at init_theta
 * Has additional parameters with precomputed matrix multiplies
 * Returns the optimal theta as arma::colvec
 * Also updates the values of teh reference 'loglik' to the maixmum loglikelihood value
 */
arma::colvec newton_raphson (const arma::colvec& init_theta, const arma::colvec& y_in, const arma::mat& V, const std::vector<int>& groups, int num_groups, double mu, double sigma, double& loglik, bool sm_inv, bool verbose){
  // initial theta
  arma::colvec theta = init_theta;
  arma::mat G = sher_morr_inv(V, theta, groups, sm_inv);
  double ll;
  arma::colvec V1 = sum (V, 1);

  // make the initial y compuations
  arma::colvec Vy = trans(V)*y_in;
  arma::colvec y = y_in - mu;
  arma::mat y_hat = compute_y_hat (y, V, groups, num_groups);
  arma::colvec y_hat_y = trans(y_hat)*y;
  arma::mat y_hat_y_hat = trans(y_hat)*y_hat;

  // loop until maximum iterations
  for (int iter = 0; iter < MAXIT; iter++){
    if (verbose){
      Rcout << "----------------------" << arma::endl;
      Rcout << "Iteration: " << iter + 1 << arma::endl;
    }

    // matrix inverse at current theta
    arma::mat GV = G*V;

    // likelihood computations
    loglikelihood (ll, theta, y, y_hat, G, sigma);
    arma::colvec grad = gradient (theta, GV, V, groups, y_hat_y, y_hat_y_hat, sigma);
    arma::mat hess = hessian (GV, V, groups, y_hat_y_hat, sigma);

    arma::colvec step = arma::solve (hess, grad);
    double decrement = dot (grad, step);
    if(verbose){
      Rcout << "Loglik: " << ll << arma::endl;
      Rcout << "Decrement: " << decrement << arma::endl;
    }
    if (decrement < TOL){
      break;
    }

    // backtrack
    double t = 1; // backtrackong parameter
    arma::colvec new_theta = theta;
    arma::mat new_G;

    double new_ll;
    while (true){
      new_theta = theta - t*step;
      new_G = sher_morr_inv(V, new_theta, groups, sm_inv);
      bool is_pos_def_cone = loglikelihood(new_ll, new_theta, y, y_hat, new_G, sigma);
      
      if (verbose){
	Rcout << "\tBacktrack!" << arma::endl;
	Rcout << "\t\tt: " << t << arma::endl;
	Rcout << "\t\tLikelihood: " << ll << arma::endl;
	Rcout << "\t\tNew likelihood: " << new_ll << arma::endl;
	Rcout << "\t\tTarget: " << ll - ALPHA*decrement << arma::endl;
      }

      if (is_pos_def_cone){ // in the cone, so check objective decrease
	if (new_ll <= ll - ALPHA*t*decrement){
	  break;
	}
      }
      t *= BETA;
    }
    theta = new_theta;
    G = new_G;
  }
  loglik = -ll;
  return theta;
}


arma::colvec biconvex_newton_raphson(const arma::colvec& init_theta, double& mu, const arma::colvec& y, const arma::mat& V, const std::vector<int>& groups, int num_groups, double sigma, double& loglik, bool sm_inv, bool verbose){
  double loglik_local=-1.0/0.0, mu_local = 0, old_loglik;
  arma::colvec theta_local = init_theta;

  // initial compuations
  arma::colvec Vy = trans(V)*y;
  arma::colvec V1 = sum(V, 1);
  
  // iterate updates until convergence
  int iter = 1;
  while (true){
    if (verbose){
      Rcout << "-----------------------------------" << arma::endl;
      Rcout << "Biconvex iteration: " << iter << arma::endl;
      Rcout << "\t mu = " << mu_local << arma::endl;
    }
    old_loglik = loglik_local;
    theta_local = newton_raphson(theta_local, y, V, groups, num_groups, mu_local, sigma, loglik_local, sm_inv, false);
    if (verbose){
      Rcout << "\told loglik: " << old_loglik << arma::endl;
      Rcout << "\tnew loglik: " << loglik_local << arma::endl;
      Rcout << "\t change: " << std::fabs(loglik_local - old_loglik) << arma::endl;
    }
    if (std::fabs(old_loglik - loglik_local) < 0.000000001) break;
    mu_local = update_mu_elr (theta_local, y, V, Vy, V1, groups);

    iter++;
  }
  
  // output
  loglik = loglik_local;
  mu = mu_local;
  return theta_local;
}


/*
 * Exported function for maximising the prototype likelihood
 * Using the lasso projection matrices
 * Accepts an initial theta, multiple responses y (one in each column), matrix of projection matrix eigenvlaues V, 
 * a labelling of the groups of the columns of V and the (known) sigma at which we optimise
 * Returns a matrix with the same number of columns as y (i.e. one for each replciation of the response)
 * First row of returned matrix is the sequence of maximised loglikelihood values
 * Next few rows (same as the number of rows in init_theta) gives the sequence of theta estimates
 */
// [[Rcpp::export]]
NumericMatrix rcpp_maximise_likelihood (const NumericMatrix& init_theta, const NumericMatrix& y, const NumericMatrix& V, const IntegerVector& groups, const NumericVector&  mu, double sigma, bool sm_inv){
  
  // ingest the data
  arma::mat init_theta_arma = Rcpp::as<arma::mat>(init_theta);
  arma::mat y_arma = Rcpp::as<arma::mat>(y);
  arma::mat V_arma = Rcpp::as<arma::mat>(V);

  // space for output
  int cols = y_arma.n_cols;
  NumericVector ll_out(cols);
  NumericVector mu_out(cols);
  arma::mat theta_out(init_theta_arma.n_rows, cols, arma::fill::zeros);

  // fix groups
  int num_groups;
  std::vector<int> fixed_groups = fix_groups(groups, num_groups);

  for (int c = 0; c < cols; c++){
    arma::colvec current_y = y_arma.col(c);
    arma::colvec current_init_theta = init_theta_arma.col(c);
    
    // Newton-Raphson
    double loglik, mu_local=mu[c];
    arma::colvec theta;
    if (std::isinf(mu[c])){
      theta = biconvex_newton_raphson(current_init_theta, mu_local, current_y, V_arma, fixed_groups, num_groups, sigma, loglik, sm_inv, false);
    }else{
      theta = newton_raphson(current_init_theta, current_y, V_arma, fixed_groups, num_groups, mu[c], sigma, loglik, sm_inv, false);
    }
    ll_out[c] = loglik;
    theta_out.col(c) = theta;
    mu_out[c] = mu_local;
  }

  arma::mat out = join_cols(join_cols(Rcpp::as<arma::rowvec>(ll_out), theta_out), Rcpp::as<arma::rowvec>(mu_out));
  return Rcpp::wrap(out);
}

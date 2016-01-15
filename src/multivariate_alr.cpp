#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

#include <math.h>
#include <cmath>
#include <vector>
#include <map>
using namespace std;



/*
 * Fixs the the group indices and gets an overall count
 */
std::vector<int> fix_groups (const NumericVector& groups, int& num_groups){
  int length = groups.size();
  std::vector<int> fixed_groups; // will hold the fixed groups
  std::map<int, int> group_index_map; // mapping from old group label to fixed ones
  int current_group = 0;
  
  for (int i = 0; i < length; i++){
    int group = groups[i];
    std::map<int, int>::iterator it = group_index_map.find(group);
    if (it == group_index_map.end()){ // haven't seen this group yet
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
 * Computes the g vector (used in computation of LR stat)
 */
arma::colvec compute_g (const std::vector<int>& groups, int num_groups){
  arma::colvec g(num_groups, arma::fill::zeros);
  for (std::vector<int>::const_iterator it = groups.begin(); it != groups.end(); it++){
    g[*it]++;
  }
  return g;
}


/*
 * Computes the H matrix (used in computation of LR stat)
 */
arma::mat compute_H (const arma::mat& V, const std::vector<int>& groups, int num_groups){
  arma::mat H (num_groups, num_groups, arma::fill::zeros);
  for (int k = 0; k < groups.size(); k++){
    int group_k = groups[k];
    for (int l = k; l < groups.size(); l++){
      int group_l = groups[l];
      double dot = arma::dot(V.col(k), V.col(l));
      double dot_sq = dot*dot;

      H.col(group_l)[group_k] += dot_sq;
      if (group_k != group_l){
	H.col(group_k)[group_l] += dot_sq;
      }
    }
  }

  return H;
}


/*
 * Computes a single replicate of the approximate LR stat
 */
arma::colvec update_theta(const arma::mat& V, const arma::colvec& y, const arma::mat& H, const arma::colvec& g, const std::vector<int>& groups, double sigma, double& loglik){
  int num_groups = g.n_rows;
  int num_eigenvecs = V.n_cols;
  int n = y.n_rows;
  
  arma::colvec Vy = trans(V)*y;
  arma::mat y_hat(n, num_groups, arma::fill::zeros);
  for (int k = 0; k < num_eigenvecs; k++){
    y_hat.col(groups[k]) += Vy[k]*V.col(k);
  }

  // first compute y_hat_y
  arma::colvec y_hat_y = trans(y_hat)*y;
  arma::mat y_hat_y_hat = trans(y_hat)*y_hat;
  //arma::colvec y_hat_y(num_groups, arma::fill::zeros);
  //for (int k = 0; k < num_eigenvecs; k++){
  // double current = Vy[k];
  // y_hat_y[groups[k]] += current*current;
  //}

  // now for y_hat_y_hat
  //arma::mat y_hat_y_hat(num_groups, num_groups, arma::fill::zeros);
  //for (int k = 0; k < num_eigenvecs; k++){
  //  for (int l = k; l < num_eigenvecs; l++){
  //   y_hat_y_hat.col(groups[l])[groups[k]] += Vy[k]*Vy[l]*VV.col(k)[l];
  // }
  //}

  // and compute the statistic
  arma::colvec v = y_hat_y/sigma/sigma - g;
  arma::mat M = H + y_hat_y_hat/sigma/sigma;

  arma::colvec theta = arma::solve (M, v);
  loglik = 0.5*dot(v, theta);
  return  arma::solve(M, v);
}


/*
 * Single update of the mu parameter, given the value of theta
 */
double update_mu (const arma::colvec& y, const arma::colvec& theta, const arma::mat& V, const arma::colvec& Vy, const arma::colvec& V1, const std::vector<int>& groups){
  // data size
  int n = y.n_rows;
  int num_cols = V.n_cols;

  // find Gy and G1
  arma::colvec Gy = y;
  arma::colvec G1(n, arma::fill::ones);
  for (int c = 0; c < num_cols; c++){
    Gy -= theta[groups[c]]*Vy[c]*V.col(c);
    G1 -= theta[groups[c]]*V1[c]*V.col(c);
  }

  return dot(Gy, G1)/dot(G1, G1);
}


/*
 * Iterates theta and mu updates until convergence of loglikelihood
 * Returns the final theta vector
 * Fills the references for mu and loglik with the final values of these quantities -- passes back to calling function
 */
arma::colvec biconvex_updates(const arma::mat& V, const arma::colvec& Vy, const arma::colvec& V1, const arma::colvec& y, const arma::mat& H, const arma::colvec& g, const std::vector<int>& groups, double sigma, double& mu, double& loglik){
  double mu_local = mu; // initial mu
  arma::colvec theta;
  double loglik_local=1.0/0.0, old_loglik;
  
  // alternate updates until convergence
  while (true){
    arma::colvec current_z = y - mu_local;
    old_loglik = loglik_local;
    theta = update_theta(V, current_z, H, g, groups, sigma, loglik_local); // update theta
    if (std::fabs(loglik_local - old_loglik) < 0.000000001) break;
    mu_local = update_mu(y, theta, V, Vy, V1, groups);
  }
  
  // output
  mu = mu_local;
  loglik = loglik_local;
  return theta;
}


/*
 * Computes the approximate LR stat for the replcates of y (each a column of y_mat)
 * V is the matrix of eigenvectors of the projection matrices; groups the group of each column in V
 */
//[[Rcpp::export]]
NumericMatrix rcpp_maximise_approx_likelihood(const NumericMatrix& y_mat, const NumericMatrix& V, const NumericVector& groups, const NumericVector& mu, double sigma){
  // ingest the data
  arma::mat y_arma = Rcpp::as<arma::mat>(y_mat);
  arma::mat V_arma = Rcpp::as<arma::mat>(V);
  int num_replicates = y_arma.n_cols;

  // fix and count groups
  int num_groups;
  std::vector<int> fixed_groups = fix_groups (groups, num_groups);

  // compute g, H
  arma::colvec g = compute_g(fixed_groups, num_groups);
  arma::mat H = compute_H(V_arma, fixed_groups, num_groups);

  // space for output
  NumericVector ll_out(num_replicates);
  NumericVector mu_out(num_replicates);
  arma::mat theta_out (num_groups, num_replicates, arma::fill::zeros);

  // find maximum approx likelihood for each replicate
  for (int r = 0; r < num_replicates; r++){
    double mu_local = mu[r];
    arma::colvec theta_local;
    double loglik;
    
    // check whether the current mu needs to be estimated or not
    if (std::isinf(mu[r])){ // biconvex
      mu_local = 0; // initial mu is 0
      arma::colvec current_y = y_arma.col(r);
      arma::colvec Vy = trans(V_arma)*current_y;
      arma::colvec V1 = sum(V_arma, 1);

      theta_local = biconvex_updates(V_arma, Vy, V1, current_y, H, g, fixed_groups, sigma, mu_local, loglik);
    }else{ // single theta update
      arma::colvec current_z = y_arma.col(r) - mu[r];
      theta_local = update_theta(V_arma, current_z, H, g, fixed_groups, sigma, loglik);
    }
    theta_out.col(r) = theta_local;
    mu_out[r] = mu_local;
    ll_out[r] = loglik;
  }
  
  arma::mat out = join_cols(join_cols (Rcpp::as<arma::rowvec>(ll_out), theta_out), Rcpp::as<arma::rowvec>(mu_out));
  return Rcpp::wrap(out);
}

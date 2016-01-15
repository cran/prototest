#include "RcppArmadillo.h"
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

#include <cmath>
using namespace std;

/*
 * Computes the exact likelihood, given some precomputed quantities from the data.
 * Likelihood evaluated at parameter values 'theta', 'mu' and 'sigma' (all assumed finite)
 * data includes the response y, its projection on the columns of interest (Py) and the number of columns onto which we project (M)
 */
double compute_exact_lr (double eta, double mu, double sigma, const arma::colvec& y, const arma::colvec& Py, const arma::colvec& P1, int M){
  int n = y.n_rows;
  
  arma::colvec z = y - mu;
  return 0.5*M*log(1-eta) - n*log(sigma) - 0.5*dot(z, z)/sigma/sigma + 0.5*eta*dot(z, Py - mu*P1)/sigma/sigma;
}

/*
 * Computes the approximate likelihood, given some precomputed quantities from the data.
 * Likelihood evaluated at the parameter values 'theta', 'sigma' and 'mu'.
 * Data includes the response (y), its projection onto the selcted columns (Py) and the number of selected columns (M)
 */
double compute_approx_lr (double eta, double mu, double sigma, const arma::colvec& y, const arma::colvec& Py, const arma::colvec& P1, int M){
  int n = y.n_rows;

  arma::colvec z = y - mu;
  return -0.5*M*eta - 0.25*M*eta*eta - n*log(sigma) - 0.5*dot(z, z)/sigma/sigma + 0.5*eta*dot(z, Py - mu*P1)/sigma/sigma;
}

/*
 * Update functions
 */
double update_eta (double mu, double sigma, double yPy, double yP1, double oneP1, int M){
  return 1 - M*sigma*sigma/(yPy + oneP1*mu*mu - 2*mu*yP1);
  //return 1 - mu*yP1/2/yPy - sqrt(mu*mu*yP1*yP1 + 4*sigma*sigma*M*yPy)/2/yPy;
}
double update_eta_approx (double mu, double sigma, double yPy, double yP1, double oneP1, int M){
  return (yPy + oneP1*mu*mu - 2*mu*yP1)/M/sigma/sigma - 1;
  //return (yPy - yP1*mu - sigma*sigma*M)/(yPy + sigma*sigma*M);
}
double update_mu (double eta, double y1, double yP1, double oneP1, int n){
  return (y1 - eta*yP1)/(n - eta*oneP1);
  //return (y1 - theta*yP1)/n;
}
double update_sigma (double eta, double mu, const arma::colvec& y, const arma::colvec& Py, const arma::colvec& P1){
  arma::colvec z = y - mu;
  int n = y.n_rows;
  return sqrt((dot(z, z) - eta*dot(z, Py - mu*P1))/n);
}


/*
 * Workhorse function.
 * Iterates update formulas for mu, sigma and theta (in that order), with initial theta = 0
 * Occurs until convergence (i.e. none of the three parameters change much over a single iteration)
 * Returns the maximised loglikelihood and populates the reference parameters with the appropriate theta, mu and sigma values
 */
double maximise_lr (const arma::colvec& y, const arma::mat& U, double& eta, double& mu, double& sigma, bool exact, double tol, int maxit){
  // precompute some quantities
  int n = y.n_rows;
  int M = U.n_cols;
  arma::colvec Uy = trans(U)*y;
  double y1 = sum(y);
  arma::colvec Py = U*Uy;
  arma::colvec P1 = U*trans(sum (U, 0));
  double yPy = dot(Uy, Uy);
  double yP1 = sum (Py);
  double oneP1 = sum(P1);

  // save original parameter values
  double orig_eta = eta;
  double orig_mu = mu;
  double orig_sigma = sigma;

  // iterate the updating formulae
  double eta_local = 0, mu_local = mu, sigma_local = sigma;
  int iter = 0;
  while (true){
    double eta_change = 0, mu_change = 0, sigma_change = 0;
    
    // update mu
    if (!std::isfinite(orig_mu)){
      double new_mu = update_mu(eta_local, y1, yP1, oneP1, n);
      mu_change = fabs(mu_local - new_mu);
      mu_local = new_mu;
    }

    // update sigma
    if (!std::isfinite(orig_sigma)){
      double new_sigma = update_sigma (eta_local, mu_local, y, Py, P1);
      sigma_change = fabs(sigma_local - new_sigma);
      sigma_local = new_sigma;
    }

    // update theta
    if (!std::isfinite(orig_eta)){
      double new_eta = exact ? update_eta(mu_local, sigma_local, yPy, yP1, oneP1, M) : update_eta_approx (mu_local, sigma_local, yPy, yP1, oneP1, M);
      eta_change = fabs(eta_local - new_eta);
      eta_local = new_eta;
    }

    // check for convergence
    if (mu_change <= tol && sigma_change <= tol) break;

    // iterations
    iter++;
    if (iter >= maxit) break;
  }

  double loglik = exact ? compute_exact_lr (eta_local, mu_local, sigma_local, y, Py, P1, M) : compute_approx_lr(eta_local, mu_local, sigma_local, y, Py, P1, M);

  // update parameter slots
  eta = eta_local;
  mu = mu_local;
  sigma = sigma_local;

  return loglik;
}


//[[Rcpp::export]]
NumericVector rcpp_compute_lr_stat (const NumericMatrix& U, const NumericMatrix& y, double mu, double sigma, bool exact, bool verbose, double tol, int maxit){
  // ingest input
  arma::mat U_arma = Rcpp::as<arma::mat>(U);
  arma::mat y_arma = Rcpp::as<arma::mat>(y);
  int num_samples = y_arma.n_cols;

  // ready for output
  NumericVector ll_stats(num_samples);
  for (int s = 0; s < num_samples; s++){
    double eta = 1.0/0.0;
    double max_ll = maximise_lr (y_arma.col(s), U_arma, eta, mu, sigma, exact, tol, maxit);
    eta = 0;
    double max_ll_0 = maximise_lr (y_arma.col(s), U_arma, eta, mu, sigma, exact, tol, maxit);

    ll_stats[s] = 2*(max_ll-max_ll_0);
  }
  return ll_stats;
}

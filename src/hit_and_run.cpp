#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

#include <math.h>
#include <cmath>
using namespace std;

#define LARGE 1000

/*
 * Generates a uniform RV replicate between a and b
 */
double generate_unif(double a, double b){
  //double u = ((double)(rand()) + 1)/((double)(RAND_MAX) + 1);
  double u = unif_rand();
  return (b-a)*u + a;
}


/*
 * Generates a single replicate of the truncated (below at a) exponential with scale paramater lambda 
 */
double generate_exp (double a, double lambda){
  double u = generate_unif (0, 1);
  return a - log(u)/lambda;
}


/*
 * Generate a single N(0,1) replicate using the Box Muller transformation
 */
double generate_std_normal(){
  double u1 = generate_unif (0,1);
  double u2 = generate_unif (0,1);

  return cos(2*M_PI*u1)*sqrt(-2*log(u2));
}


/*
 * Generates n independent N(0, 1) replicates using the Box Muller transformations
 */
arma::colvec generate_std_normal_sample(int n){
  arma::colvec v (n);
  for (int s = 0; s < n; s++){
    v(s) = generate_std_normal();
  }
  return v;
}


/*
 * Generates a single accept-reject replication for the two-sided truncated normal RV
 * using the uniform reference distribution between a and b
 * as in 'Simulation of truncated normal variables' of Robert
 */
double trunc_normal_accept_reject_unif (double a, double b){
  while (true){ // loop until we accept
    double z = generate_unif (a,b);
    double u = generate_unif (0,1);
    double rho;
    
    // do the checks
    if (b < 0){
      rho = exp (0.5*(b*b - z*z));
    }else if (a > 0){
      rho = exp (0.5*(a*a - z*z));
    }else{
      rho = exp (-0.5*z*z);
    }

    // accept or reject?
    if (u <= rho) return z;
  }
}

/*
 * Generates a single replicate from a truncated normal distribution.
 * No limit above; NEGATIVE limit below (i.e a < 0).
 * Just sample normal RV until z > a
 */
double trunc_normal_accept_reject_simple (double a){
  while (true){
    double z = generate_std_normal();
    if (z >= a) return z;
  }
}

/*
 * Generates a single replicate from a truncated normal distribution.
 * No limit above; lower limit is POSITIVE (a > 0).
 * Uses the exponential dist reference as in Robert
 */
double trunc_normal_accept_reject_exp (double a){
  double a_star = 0.5*(a + sqrt(a*a + 4));
  
  while (true){
    double z = generate_exp (a_star, a);
    double u = generate_unif (0,1);
    double rho = exp(-0.5*(z-a_star)*(z-a_star));
    if (u <= rho) return z;
  }
}

/*
 * Generates a single truncated normal replicate between a and b.
 * Mean param 0; sd param 1
 */
double generate_trunc_normal (double a, double b){
  // distinguish cases
  //if (isinf(a) && isinf(b)){ // no restriction -- generate a N(0,1)
  if (fabs(a) > LARGE && fabs(b) > LARGE){
    return generate_std_normal();
  }
  if (fabs(b) > LARGE){ // no limit above
    if (a <= 0) return trunc_normal_accept_reject_simple(a);
    else return trunc_normal_accept_reject_exp(a);
  }
  if (fabs(a) > LARGE){ // no limit below (mirror image of no limit above)
    if (b >= 0) return -trunc_normal_accept_reject_simple(-b);
    else return -trunc_normal_accept_reject_exp(-b);
  }

  // if we get here, we have limits on both sides, so do the accept-reject method
  return trunc_normal_accept_reject_unif(a, b);
}


/*
 * Computes the limits (vm and vp) of the polytope desceibed bt A and b.
 * in the direction of z, starting at y_tilde
 * Returns nothing; populates vp and vm
 */
void compute_vp_vm (double& vp, double& vm, const arma::colvec& y, const arma::colvec& eta, const arma::mat& A, const arma::colvec& b){
  double vm_loc = -INFINITY;
  double vp_loc = INFINITY;
  arma::colvec c = eta/dot(eta, eta);
  arma::colvec z = y - dot(y, eta)*c;

  arma::colvec Az = A*z;
  arma::colvec Ac = A*c;

  for (int i = 0; i < Ac.n_rows; i++){
    if (Ac[i] > 0){ // might change the positive bound                                                                         
      double current = (b[i] - Az[i])/Ac[i];
      if (current < vp_loc){
	vp_loc = current;
      }
    }
    if (Ac[i] < 0){ // might change the negative bound                                                                         
      double current = (b[i] - Az[i])/Ac[i];
      if (current > vm_loc){
	vm_loc = current;
      }
    }
  }  

  vp = vp_loc;
  vm = vm_loc;
}



/*
 * Generates hit and run samples (after a user-defined burn in)
 * Samples are multivariate Gaussian with mean 0 and covariance matrix I
 * Confined to the polytope Ay <= b
 */
// will remove export
// [[Rcpp::export]]
NumericMatrix rcpp_generate_hit_and_run_samples(int num_samples, int burn_in, const NumericVector& init_y, const NumericMatrix& A, const NumericVector& b){
  // ingest data
  arma::colvec y_tilde = Rcpp::as<arma::colvec>(init_y);
  arma::mat A_arma = Rcpp::as<arma::mat>(A);
  arma::colvec b_arma = Rcpp::as<arma::colvec>(b);

  int n = y_tilde.n_rows;
  int total_samples = num_samples + burn_in;

  // generate samples
  GetRNGstate();
  arma::mat samples(n, num_samples, arma::fill::zeros);
  for (int s = 0; s < total_samples; s++){
    //Rcout << "\t" << s;
    // random direction
    arma::colvec z = generate_std_normal_sample(n);
    z = z/sqrt(dot(z,z));

    // find polytope bounds in the direction of z
    double vm, vp;
    compute_vp_vm (vp, vm, y_tilde, z, A_arma, b_arma);
    //Rcout << "Vp " << vp << " Vm " << vm << arma::endl;

    // generate truncated Gaussian replicate
    double K = generate_trunc_normal (vm, vp);
    //Rcout << "K " << K << arma::endl;

    // update y_tilde and store
    y_tilde = y_tilde + (K - dot(y_tilde, z))*z;
    if (s >= burn_in){
      samples.col(s-burn_in) = y_tilde;
    }
  }
  PutRNGstate();
  return Rcpp::wrap(samples);
}

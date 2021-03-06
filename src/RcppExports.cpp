// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// rcpp_generate_hit_and_run_samples
NumericMatrix rcpp_generate_hit_and_run_samples(int num_samples, int burn_in, const NumericVector& init_y, const NumericMatrix& A, const NumericVector& b);
RcppExport SEXP prototest_rcpp_generate_hit_and_run_samples(SEXP num_samplesSEXP, SEXP burn_inSEXP, SEXP init_ySEXP, SEXP ASEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< int >::type num_samples(num_samplesSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type init_y(init_ySEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type b(bSEXP);
    __result = Rcpp::wrap(rcpp_generate_hit_and_run_samples(num_samples, burn_in, init_y, A, b));
    return __result;
END_RCPP
}
// rcpp_maximise_approx_likelihood
NumericMatrix rcpp_maximise_approx_likelihood(const NumericMatrix& y_mat, const NumericMatrix& V, const NumericVector& groups, const NumericVector& mu, double sigma);
RcppExport SEXP prototest_rcpp_maximise_approx_likelihood(SEXP y_matSEXP, SEXP VSEXP, SEXP groupsSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type y_mat(y_matSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type V(VSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    __result = Rcpp::wrap(rcpp_maximise_approx_likelihood(y_mat, V, groups, mu, sigma));
    return __result;
END_RCPP
}
// rcpp_maximise_likelihood
NumericMatrix rcpp_maximise_likelihood(const NumericMatrix& init_theta, const NumericMatrix& y, const NumericMatrix& V, const IntegerVector& groups, const NumericVector& mu, double sigma, bool sm_inv);
RcppExport SEXP prototest_rcpp_maximise_likelihood(SEXP init_thetaSEXP, SEXP ySEXP, SEXP VSEXP, SEXP groupsSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP sm_invSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type init_theta(init_thetaSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type V(VSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< bool >::type sm_inv(sm_invSEXP);
    __result = Rcpp::wrap(rcpp_maximise_likelihood(init_theta, y, V, groups, mu, sigma, sm_inv));
    return __result;
END_RCPP
}
// rcpp_compute_lr_stat
NumericVector rcpp_compute_lr_stat(const NumericMatrix& U, const NumericMatrix& y, double mu, double sigma, bool exact, bool verbose, double tol, int maxit);
RcppExport SEXP prototest_rcpp_compute_lr_stat(SEXP USEXP, SEXP ySEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP exactSEXP, SEXP verboseSEXP, SEXP tolSEXP, SEXP maxitSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type U(USEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< bool >::type exact(exactSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type maxit(maxitSEXP);
    __result = Rcpp::wrap(rcpp_compute_lr_stat(U, y, mu, sigma, exact, verbose, tol, maxit));
    return __result;
END_RCPP
}

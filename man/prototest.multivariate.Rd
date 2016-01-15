\name{prototest.multivariate}
\alias{prototest.multivariate}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Perform Prototype or F tests for Significance of Groups of Predictors in the Multivariate Model
}
\description{
Perform prototype or F tests for significance of groups of predictors in the multivariate model. Choose either exact or approximate likelihood ratio prototype tests (ELR) or (ALR) or F test or marginal screening prototype test. Options for selective or non-selective tests. Further options for non-sampling or hit-and-run reference distributions for selective tests. 
}
\usage{
prototest.multivariate(x, y, groups, test.group, type = c("ELR", "ALR", "F", "MS"), 
selected.col = NULL, lambda, mu = NULL, sigma = 1, 
hr.iter = 50000, hr.burn.in = 5000, verbose = FALSE, tol = 10^-8)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
input matrix of dimension \emph{n}-by-\emph{p}, where \emph{p} is the number of predictors over all predictor groups of interest. Will be mean centered and standardised before tests are performed.
}
  \item{y}{
response variable. Vector of length \emph{n}, assumed to be quantitative.
}
  \item{groups}{
group membership of the columns of \code{x}. Vector of length \emph{p}, which each element containing the goup label of the corresponding column in \code{x}.
}
  \item{test.group}{
group label for which we test nullity. Should be one of the values seen in \code{groups}. See Details for further explanation.
}
  \item{type}{
type of test to be performed. Can select one at a time. Options include the exact and approximate likelihood ratio prototype tests of Reid et al (2015) (ELR, ALR), the F test and the marginal screening prototype test of Reid and Tibshirani (2015) (MS). Default is ELR.
}
  \item{selected.col}{
preselected columns selected by the user. Vector of indices in the set \{1, 2, ... \emph{p}\}. Used in conjunction with \code{groups} to ascertain for which groups the user has specified selected columns. Should it find any selected columns within a group, no further action is taken to select columns. Should no columns within a group be specified, columns are selected using either lasso or the marginal screening procedure, depending on the test. If all groups have prespecified columns, a non-selective test is performed, using the classical distributional assumptions (exact and/or asymptotic) for the test in question. If any selection is performed, selective tests are performed. Default is \code{NULL}, requiring the selection of columns in all the groups. 
}
  \item{lambda}{
regularisation parameter for the lasso fit. Same for each group. Must be supplied when at least one group has unspecified columns in \code{selected.col}. Will be supplied to \code{glmnet}. This is the unstandardised version, equivalent to \code{lambda}/\code{n} supplied to \code{glmnet}.
}
  \item{mu}{
mean parameter for the response. See Details below. If supplied, it is first subtracted from the response to yield a zero-mean (at the population level) vector for which we proceed with testing. If \code{NULL} (the default), this parameter is treated as nuisance parameter and accounted for as such in testing.
}
  \item{sigma}{
error standard deviation for the response. See Details below. Must be supplied. If not, it is assumed to be 1. Required for computation of some of the test statistics.
}
  \item{hr.iter}{
number of hit-and-run samples required in the reference distribution of the a selective test. Applies only if \code{selected.col} is \code{NULL}. Default is 50000. Since dependent samples are generated, large values are required to generate good reference distributions. If set to 0, the function tries to applu a non-sampling selective test (provided \code{selected.col} is \code{NULL}), if possible. If non-sampling test is not possible, the function exits with a message.
}
  \item{hr.burn.in}{
number of burn-in hit-and-run samples. These are generated first so as to make subsequent hit-and-run realisations less dependent on the observed response. Samples are then discarded and do not inform the null reference distribution.
}
  \item{verbose}{
should progress be printed?
}
  \item{tol}{
convergence threshold for iterative optimisation procedures.
}
}
\details{
The model underpinning each of the tests is \deqn{
	y = \mu + \sum_{k = 1}^K \theta_k\cdot\hat{y}_k + \epsilon}{\emph{y = mu + sum_k theta_k hat_y_k + epsilon}}
where \eqn{\epsilon \sim N(0, \sigma^2I)}{\emph{epsilon} is Gaussian with mean 0 and variance sigma^2} and \emph{K} is the number of predictor groups. \eqn{\hat{y}_k}{\emph{y_hat_k}} depends on the particular test considered.

In particular, for the ELR, ALR and F tests, we have \eqn{\hat{y}_k = P_{M_k}\left(y-\mu\right)}{\emph{y_hat_k = P_M_k(y - mu)}}, where \eqn{P_{M_k} = X_{M_k}\left(X_{M_k}^\top X_{M_k}\right)^{-1}X_{M_k}^\top}{\emph{P_M_k = X_M_kX_M_k^dagger}}. \eqn{X_M}{\emph{X_M}} is the input matrix reduced to the columns with indices in the set \emph{M}. \eqn{M_k}{\emph{M_k}} is the set of indices selected from considering group \emph{k} of predictors in isolation. This set is either provided by the user (via \code{selected.col}) or is selected automatically (if \code{selected.col} is \code{NULL}). If the former, a non-selective test is performed; if the latter, a selective test is performed, with the restrictions \eqn{Ay \leq b}{\emph{Ay <= b}}, as set out in Lee et al (2015) and stacked as in Reid and Tibshirani (2015). 

For the marginal screening prototype (MS) test, \eqn{\hat{y}_k = x_{j^*}}{\emph{y_hat_k = x_j_star}} where \eqn{x_j}{\emph{x_j}} is the \eqn{j^{th}}{\emph{jth}} column of \code{x} and \eqn{j^* = {\rm argmax}_{j \in C_k} |x_j^\top y|}{is the column of maximal marginal correlation with the response in set \emph{C_k}}, where \eqn{C_k}{\emph{C_k}} is the set of indices in the overall predictor set corresponding to predictors in the \eqn{k^{th}}{\emph{kth}} group.

All tests test the null hypothesis \eqn{H_0: \theta_{k^*} = 0}{H_0: \emph{theta_k_star = 0}}, where \eqn{k^*}{\emph{k_star}} is supplied by the user via \code{test.group}. Details of each are described in Reid et al (2015). 

}

\value{
A list with the following four components:
\item{ts}{The value of the test statistic on the observed data.}
\item{p.val}{Valid p-value of the test.}
\item{selected.col}{Vector with columns selected for prototype formation in the test. If initially \code{NULL}, this will now contain indices of columns selected by the automatic column selection procedures of the test.}
\item{y.hr}{Matrix with hit-and-run replications of the response. If sampled selective test was not performed, this will be \code{NULL}.}
}
\references{
Reid, S. and Tibshirani, R. (2015) \emph{Sparse regression and marginal testing using cluster prototypes}. \url{http://arxiv.org/pdf/1503.00334v2.pdf}. \emph{Biostatistics \doi{10.1093/biostatistics/kxv049}}\cr
Reid, S., Taylor, J. and Tibshirani, R. (2015) \emph{A general framework for estimation and inference from clusters of features}. Available online: \url{http://arxiv.org/abs/1511.07839}.
}
\author{
Stephen Reid
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{prototest.univariate}}
}
\examples{
require (prototest)

### generate data
set.seed (12345)
n = 100
p = 80

X = matrix (rnorm(n*p, 0, 1), ncol=p)


beta = rep(0, p)
beta[1:3] = 0.1 # three signal variables: number 1, 2, 3
signal = apply(X, 1, function(col){sum(beta*col)})
intercept = 3

y = intercept + signal + rnorm (n, 0, 1)

### treat all columns as if in same group and test for signal

# non-selective ELR test with nuisance intercept
elr = prototest.univariate (X, y, "ELR", selected.col=1:5)
# selective F test with nuisance intercept; non-sampling
f.test = prototest.univariate (X, y, "F", lambda=0.01, hr.iter=0) 
print (elr)
print (f.test)

### assume variables occur in 4 equally sized groups
num.groups = 4
groups = rep (1:num.groups, each=p/num.groups)

# selective ALR test -- select columns 21-25 in 2nd group; test for signal in 1st; hit-and-run
alr = prototest.multivariate(X, y, groups, 1, "ALR", 21:25, lambda=0.005, hr.iter=20000)
# non-selective MS test -- specify first column in each group; test for signal in 1st
ms = prototest.multivariate(X, y, groups, 1, "MS", c(1,21,41,61)) 
print (alr)
print (ms)
}
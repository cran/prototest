\name{prototest.univariate}
\alias{prototest.univariate}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Perform Prototype or F Tests for Significance of Groups of Predictors in the Univariate Model
}
\description{
Perform prototype or F tests for significance of groups of predictors in the univariate model. Choose either exact or approximate likelihood ratio prototype tests (ELR) or (ALR) or F test or marginal screening prototype test. Options for selective or non-selective tests. Further options for non-sampling or hit-and-run null reference distributions for selective tests.
}
\usage{
prototest.univariate(x, y, type = c("ALR", "ELR", "MS", "F"), 
selected.col = NULL, lambda, mu = NULL, sigma = 1, hr.iter = 50000, 
hr.burn.in = 5000, verbose = FALSE, tol = 10^-8)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
input matrix of dimension \emph{n}-by-\emph{p}, where \emph{p} is the number of predictors in a single predetermined group of predictors. Will be mean centered and standardised before tests are performed. 
}
  \item{y}{
response variable. Vector of length emph{n}, assumed to be quantitative.
}
  \item{type}{
type of test to be performed. Can only select one at a time. Options include the exact and approximate likelihood ratio prototype tests of Reid et al (2015) (ELR, ALR), the F test and the marginal screening prototype test of Reid and Tibshirani (2015) (MS). Default is ELR.
}
  \item{selected.col}{
preselected columns specified by user. Vector of indices in the set \{1, 2, ..., \emph{p}\}. If specified, a \emph{non-selective} (classical) version of the chosen test it performed. In particular, this means the classicial \eqn{\chi^2_1}{chi-squared 1} reference distribution for the likelihood ratio tests and the F reference for the F test. Default is \code{NULL}, which directs the function to estimate the selected set with the lasso or the marginal screening procedure, depending on the test.
}
  \item{lambda}{
regularisation parameter for the lasso fit. Must be supplied when \code{selected.col} is \code{NULL}. Will be supplied to \code{glmnet}. This is the unstandardised version, equivalent to \code{lambda}/\emph{n} supplied to \code{glmnet}.
}
  \item{mu}{
mean parameter for the response. See Details below. If supplied, it is first subtracted from the response to yield a mean-zero (at the population level) vector for which we proceed with testing. If \code{NULL} (the default), this parameter is treated as nuisance parameter and accounted for as such in testing.
}
  \item{sigma}{
error standard deviation for the response. See Details below. Must be supplied. If not, it is assumed to be 1. Required for the computation of some of the test statistics.
}
  \item{hr.iter}{
number of hit-and-run samples required in the reference distrbution of a selective test. Applies only if \code{selected.col} is \code{NULL}. Default is 50000. Since dependent samples are generated, large values are required to generate good reference distributions. If set to 0, the function tries to apply a non-sampling selective test (provided \code{selected.col} is \code{NULL}), if possible. If non-sampling test is not possible, the function exits with a message.
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
The model underpinning each of the tests is
\deqn{
	y = \mu + \theta\cdot\hat{y} + \epsilon
}{\emph{y = mu + theta u_hat + epsilon}}
where \eqn{\epsilon \sim N(0, \sigma^2I)}{\emph{epsilon} is Gaussian with zero mean and variance \emph{sigma^2}} and \eqn{\hat{y}}{\emph{y_hat}} depends on the particular test considered. 

In particular, for the ELR, ALR and F tests, we have \eqn{\hat{y} = P_M\left(y - \mu\right)}{\emph{y_hat = P_M(y - mu)}}, where \eqn{P_M = X_M\left(X_M^\top X_M\right)^{-1}X_M^\top}{\emph{X_MX_M^dagger}}. \eqn{X_M}{\emph{X_M}} is the input matrix reduced to the columns in the set \emph{M}, which, in turn, is either provided by the user (via \code{selected.col}) or selected by the lasso (if \code{selected.col} is \code{NULL}). If the former, a non-selective test is performed; if the latter, a selective test is performed, with the restrictions \eqn{Ay\leq b}{\emph{Ay <= b}}, as set out in Lee et al (2015). 

For the marginal screening prototype (MS) test, \eqn{\hat{y} = x_{j^*}}{\emph{y_hat = x_j_star}} where \eqn{x_j}{\emph{x_j}} is the \eqn{j^{th}}{\emph{jth}} column of \code{x} and \eqn{j^* = {\rm argmax}_j |x_j^\top y|}{is the column of maximal marginal correlation with the response}.

All tests test the null hypothesis \eqn{H_0: \theta = 0}{H_0: \emph{theta = 0}}. Details of each are described in Reid et al (2015).
}
\value{
A list with the following four components:
\item{ts}{The value of the test statistic on the observed data.}
\item{p.val}{Valid p-value of the test.}
\item{selected.col}{Vector with columns selected. If initially \code{NULL}, this will now contain indices of columns selected by the automatic column selection procedures of the test.}
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
\code{\link{prototest.multivariate}}
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
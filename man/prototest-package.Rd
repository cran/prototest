\name{prototest-package}
\alias{prototest-package}
\alias{prototest}
\docType{package}
\title{
Inference on Prototypes from Clusters of Features
}
\description{
Procedures for testing for group-wide signal in clusters of variables. Tests can be perfromed for single groups in isolation (univariate) or multiple groups together (multivariate). Specific tests include the exact and approximate (un)selective likelihood ratio (ELR, ALR) tests described in Reid et al (2015), the selective F test and marginal screening prototype test of Reid and Tibshirani (2015). User may prespecify columns to be included in prototype formation, or allow the function to select them itself. A mixture of these two is also possible. Any variable selection is accounted for using the selective inference framework introduced in Lee et al (2013) and further developed in Lee and Taylor (2014). Options for non-sampling and hit-and-run null reference distrbutions. Tests are examples of selected model tests, a notion introduced in Fithian et al (2015).
}
\details{
\tabular{ll}{
Package: \tab prototest\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2015-11-12\cr
License: \tab GPL (>= 2)\cr
}
Only two functions provided: \code{prototest.univariate} (for tests with a single group in isolation) and \code{prototest.multivariate} (for tests with multiple groups simultaneously). Each function provides options to perform one of the ELR, ALR, F or marginal screening prototype tests. User may specify which columns are to be used in prototype construction, or leave it for the function to select. Valid tests are performed in the event of variable selection. User has option to use non-sampling null reference distributions (where available) or hit-and-run references.
}
\author{
Stephen Reid

Maintainer: Stephen Reid <sreid@stanford.edu>
}
\references{
Reid, S. and Tibshirani, R. (2015) \emph{Sparse regression and marginal testing using cluster prototypes}. \url{http://arxiv.org/pdf/1503.00334v2.pdf}. \emph{Biostatistics \doi{10.1093/biostatistics/kxv049}}\cr
Reid, S., Taylor, J. and Tibshirani, R. (2015) \emph{A general framework for estimation and inference from clusters of features}. Available online: \url{http://arxiv.org/abs/1511.07839}\cr
Lee, J.D., Sun, D.L., Sun, Y. and Taylor, J.E. (2013) \emph{Exact post-selection inference, with application to the lasso}. \url{http://arxiv.org/pdf/1311.6238v6.pdf}. \emph{Annals of Statistics (to appear)}\cr
Lee, J.D. and Taylor, J.E. (2014) \emph{Exact Post Model Selection Inference for Marginal Screening}. \url{http://arxiv.org/pdf/1402.5596v2.pdf}\cr
Fithian, W., Sun, D.L. and Taylor, J.E. (2015) \emph{Optimal Inference After Model Selection}. \url{http://arxiv.org/pdf/1410.2597v2.pdf}
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

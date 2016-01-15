\name{print.prototest}
\alias{print.prototest}
\title{
Print \code{prototest} object
}
\description{
Generic \code{print} method for \code{prototest} objects
}
\usage{
\method{print}{prototest} (x, ...)
}
\arguments{
\item{x}{object of type \code{prototest}.}
\item{...}{other parameters passed to \code{print} function.}
}
\details{
Prints the test statistic and p-value associated with the \code{prototest} object \code{x}.
}
\author{
Stephen Reid
}
\seealso{
\code{\link{prototest.univariate}}, \code{\link{prototest.multivariate}}
}
\examples{
require (prototest)

### generate data
set.seed (12345)
n = 100
p = 80

X = matrix (rnorm(n*p, 0, 1), ncol=p)


beta = rep(0, p)
beta[1:3] = 2 # three signal variables: number 1, 2, 3
signal = apply(X, 1, function(col){sum(beta*col)})
intercept = 3

y = intercept + signal + rnorm (n, 0, 1)

### treat all columns as if in same group and test for signal

# non-selective ELR test with nuisance intercept
elr = prototest.univariate (X, y, "ELR", selected.col=1:5) 
print (elr)
}
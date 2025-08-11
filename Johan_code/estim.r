# Gaussian kernel function
# INPUT
# x, y : vectors of the same length
# sig  : a positive number
# OUTPUT
# a scalar
K.gauss <- function(x, y, sig) {
  exp(-sum((x-y)^2)/(2*sig^2))
}

# one-sample U-statistic of order 2
# INPUT
# x : a matrix with n rows
# K : a function accepting two arguments, each argument being a vector
# OUTPUT
# the arithmetic mean of K(x[i,], x[j,]) over all distinct rows i and j of x
U.1 <- function(x, K) {
  n <- nrow(x)
  stopifnot(n > 1)
  U <- 0
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      U <- U + K(x[i,],x[j,])
    }
  }
  return(2 * U / (n*(n-1)))
}

# two-sample U-statistic of order (1,1)
# INPUT
# x, y : matrices with possibly different dimensions
# K : a kernel function
# OUTPUT
# the arithmetic mean of K(x[i,], y[j,]) over all rows i of x and j of y.
U.2 <- function(x, y, K) {
  m <- nrow(x)
  n <- nrow(y)
  U <- 0
  for (i in 1:m) {
    for (j in 1:n) {
      U <- U + K(x[i,], y[j,])
    }
  }
  return(U / (m*n))
}

# outer product of x and y via K; may consider function outer() instead.
# INPUT
# x, y : matrices with possibly different dimensions
# K : a kernel function
# OUTPUT
# the m-by-n matrix K(x[i,],y[j,]) over the m rows i of x and the n rows j of y
Kouter <- function(x, y, K) {
  m <- nrow(x)
  n <- nrow(y)
  res <- matrix(0, nrow = m, ncol = n)
  for (i in 1:m) {
    for (j in 1:n) {
      res[i,j] <- K(x[i,], y[j,])
    }
  }
  return(res)
}

# the two estimators
# INPUT
# x.tg, x.pl, x.mn: data matrices with the same number of columns (column vectors if only one column)
# K : kernel function
# OUTUT
# a vector of length two
# first element: the norm-based ratio estimator
# second element: the inner-product based projection estimator
pi.estim <- function(x.tg, x.pl, x.mn, K) {
  Dlt.tg.pl <- max(U.1(x.tg, K) - 2*U.2(x.tg, x.pl, K) + U.1(x.pl, K), 0)
  Dlt.tg.mn <- max(U.1(x.tg, K) - 2*U.2(x.tg, x.mn, K) + U.1(x.mn, K), 0)
  Dlt.pl.mn  <- max(U.1(x.pl,  K) - 2*U.2(x.pl,  x.mn, K) + U.1(x.mn, K), 0)
  pi.nrm <- min(sqrt(Dlt.tg.mn / Dlt.pl.mn), 1)
  pi.ipr <- min(max(((Dlt.tg.mn - Dlt.tg.pl)/Dlt.pl.mn + 1)/2, 0), 1)
  return(c(pi.nrm = pi.nrm, pi.ipr = pi.ipr))
}

# the inner-product based estimator together with plug-in estimate of its standard error
# INPUT
# x.tg, x.pl, x.mn: data matrices with the same number of columns (column vectors if only one column)
# K : kernel function
# OUTPUT
# a vector of length two
# first element: the inner-product based projection estimator
# second element: the estimated asymptotic standard error
pi.estim.var <- function(x.tg, x.pl, x.mn, K) {
  n.tg <- nrow(x.tg)
  n.pl <- nrow(x.pl)
  n.mn <- nrow(x.mn)
  
  K.tg.tg <- Kouter(x.tg, x.tg, K)
  K.pl.pl <- Kouter(x.pl, x.pl, K)
  K.mn.mn <- Kouter(x.mn, x.mn, K)

  K.tg.pl <- Kouter(x.tg, x.pl, K)
  K.tg.mn <- Kouter(x.tg, x.mn, K)
  K.pl.mn <- Kouter(x.pl, x.mn, K)
  
  U.tg.tg <- (sum(K.tg.tg) - sum(diag(K.tg.tg))) / (n.tg*(n.tg-1))
  U.mn.mn <- (sum(K.mn.mn) - sum(diag(K.mn.mn))) / (n.mn*(n.mn-1))
  U.pl.pl <- (sum(K.pl.pl) - sum(diag(K.pl.pl))) / (n.pl*(n.pl-1))
  
  U.tg.pl <- mean(K.tg.pl)
  U.tg.mn <- mean(K.tg.mn)
  U.pl.mn <- mean(K.pl.mn)
  
  Dlt.tg.pl <- U.tg.tg - 2*U.tg.pl + U.pl.pl
  Dlt.tg.mn <- U.tg.tg - 2*U.tg.mn + U.mn.mn
  Dlt.pl.mn <- U.pl.pl - 2*U.pl.mn + U.mn.mn
  
  pi.ipr <- min(max(((Dlt.tg.mn - Dlt.tg.pl)/Dlt.pl.mn + 1)/2, 0), 1)
  
  n.tg.1 <- 1/n.tg
  n.pl.1 <- 1/n.pl
  n.mn.1 <- 1/n.mn
  s <- n.tg.1 + n.pl.1 + n.mn.1
  lam.tg <- n.tg.1 / s
  lam.pl <- n.pl.1 / s
  lam.mn <- n.mn.1 / s

  tau2.tg <- var( rowMeans(K.tg.pl) - rowMeans(K.tg.mn) )
  tau2.pl <- var( colMeans(K.tg.pl) - rowMeans(K.pl.mn) )
  tau2.mn <- var( colMeans(K.tg.mn) - colMeans(K.pl.mn) )

  std.err <- sqrt(s * (lam.tg * tau2.tg + lam.pl * tau2.pl + lam.mn * tau2.mn)) / Dlt.pl.mn
  
  return(c(pi.ipr = pi.ipr, std.err = std.err))
}


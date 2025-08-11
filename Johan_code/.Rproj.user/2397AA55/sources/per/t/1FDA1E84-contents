library(mvtnorm)
source("./estim.r")
source("./asvar-gauss.r")

K <- function(x,y) {
  # function K.gauss defined in <estim.r>
  K.gauss(x, y, sig = 0.5)
}

p_source_plus <- read.table('p_source_plus.txt')
p_source_minus <- read.table('p_source_minus.txt')
p_target <- read.table('p_target.txt')

result <- pi.estim.var(p_target, p_source_plus, p_source_minus, K)

result

# pi.ipr     std.err     tau2.tg     tau2.pl     tau2.mn 
# 0.189955444 0.110299096 0.005089119 0.000626285 0.002628511 
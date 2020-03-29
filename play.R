library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
library(jsonlite)

## data <- list(mu = rep(0, 2), S = matrix(c(1, 1.98, 1.98, 4.0), ncol=2))
data <- list(mu = rep(0, 2), S = matrix(c(1, 0.0, 0.0, 1.0), ncol=2))

nchains <- 4
N <- 1000000

posterior <- stan("stanmodels/correlatedgaussian.stan",
                  data=data, chains=nchains, iter=N)
# control = list(metric="unit_e", adapt_engaged =  FALSE))

m <- monitor(posterior)

nleaps <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$n_leapfrog__})

colMeans(nleaps[2001:4000, ])

plot(1:2000, nleaps[2001:4000, 1], type="l")

acceptstats <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$accept_stat__})

plot(1:2000, acceptstats[2001:4000, 1], type="l")
colMeans(acceptstats[2001:4000, ])

stepsizes <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$stepsize})

colMeans(stepsizes[2001:4000, ])

depths <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$treedepth__
})

colMeans(depths[2001:4000, ])

plot(1:2000, depths[2001:4000, 1], type="l")

sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "adaptation_info")
})

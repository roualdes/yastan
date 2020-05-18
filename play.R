library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
library(jsonlite)
library(stringr)
library(dplyr)
library(purrr)
library(Matrix)
library(hdf5r)
library(ggplot2)
library(posterior)

posterior_d <- read_stan_csv(paste0("/Users/ez/performance-tests-cmdstan/stat_comp_benchmarks/benchmarks/arK/arK_dense/samples", 1:4, ".csv"))

posterior <- read_stan_csv(paste0("/Users/ez/performance-tests-cmdstan/stat_comp_benchmarks/benchmarks/arK/arK_diag/samples", 1:4, ".csv"))






md <- monitor(posterior_d)
ms <- monitor(posterior)

samples_d <- as.array(posterior_d)
quantile(sapply(1:100, function(i) ess_tail(samples_d[,,i])), c(0.1, 0.5, 0.9))

samples <- as.array(posterior)
quantile(sapply(1:100, function(i) ess_tail(samples[,,i])), c(0.1, 0.5, 0.9))


nchains <- 3
N <- 10000
warmup <- 2000

R <- 1
df <- data.frame(matrix(NA, nrow=1001, ncol=4))

m <- matrix(c(1, 0.0, 0.0, 4), nrow=2)
l <- lapply(1:50, function(x) m)
M <- as.matrix(bdiag(l))

data <- list(d = 100, mu = rep(0, 100), S = M)

write(toJSON(data), "stanmodels/zeromeangaussian.json")

model <- stan_model("stanmodels/zeromeangaussian.stan")

for (r in 1:R) {

    posterior <- sampling(model, data=data, chains=nchains, iter=N, warmup=warmup)

    m <- monitor(posterior)
    samples <- as.matrix(posterior)
    df[r, ] <- c(quantile(samples[,3], c(0.1, 0.5, 0.9)), m$Tail_ESS[3])
}

write.csv(df, "t2.csv", row.names = FALSE)


monitor(posterior)


posterior <- stan("stanmodels/correlatedgaussian.stan",
                  data=data, chains=nchains, iter=N)
# control = list(metric="unit_e", adapt_engaged =  FALSE))

m <- monitor(posterior)

divs <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$divergent__})

nleaps <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$n_leapfrog__})[idx, ]

plot(density(nleaps[, 1]), xlim=c(0, 10), main="Number leapfrog steps")
for (i in 2:nchains) {
    lines(density(nleaps[, i]))
}

acceptstats <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$accept_stat__})[idx,]

plot(density(acceptstats[, 1]), xlim=c(-0.1, 1.1))
for (i in 2:nchains) {
    lines(density(acceptstats[, i]))
}

stepsizes <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$stepsize})

plot(1:N, log10(stepsizes[, 1]))

depths <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "sampler_params")$treedepth__
})

mm <- sapply(1:nchains, function (x) {
    attr(posterior@sim$samples[[x]], "adaptation_info")
}) %>%
    sapply(function(x) {
        x %>%
            str_split("\n", simplify=TRUE) %>%
            .[,3] %>%
            str_remove("# ") %>%
            str_split(", ", simplify=TRUE) %>%
            as.numeric
    })

plot(density(mm[1,]), xlim = c(0, 6))
lines(density(mm[2,]))



dfh5 <- H5File$new("~/Desktop/iso.h5", mode="r")

epl <- dfh5[["isogaussian/yastan/D100/esssq"]][] / dfh5[["isogaussian/yastan/D100/leapfrog"]][]

file.h5$close_all()

ggplot(data.frame(x = epl), aes(x)) + geom_density()

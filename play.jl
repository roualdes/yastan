using Revise
using Plots
using Statistics

includet("hmc.jl")
includet("convergence.jl")
includet("models.jl")

N = 4000
ndim = 2
samples, d = stan(correlatedgaussian, N, ndim, 4);

plot(1:2000, samples[2001:end, 1, :])

plot(samples[2001:end, 1, :], samples[2001:end, 2, :],
     seriestype = :scatter, alpha = 0.1)


round.(mean(samples[2001:end, :, :], dims=[1,3]), digits=1)
round.(std(samples[2001:end, :, :], dims=[1,3]), digits=1)

map(i -> ess_bulk(samples[2001:end, i, :]), 1:ndim)
map(i -> ess_tail(samples[2001:end, i, :]), 1:ndim)
map(i -> ess_std(samples[2001:end, i, :]), 1:ndim)
map(i -> rhat(samples[2001:end, i, :]), 1:ndim)


mean(d[:leapfrog][2001:end, :], dims=1)
mean(d[:acceptstat][2001:end, :], dims=1)
mean(d[:stepsize][2001:end, :], dims=1)
mean(d[:treedepth][2001:end, :], dims=1)

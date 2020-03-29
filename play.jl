using Revise
using Plots; plotly();
using Statistics
includet("hmc.jl")
includet("convergence.jl")
includet("models.jl")

N = 1_000_000;
ndim = 2;

samples, leaps, acceptstats, εs, treedepth, energy, M =
    hmc(correlatedgaussian, N, ndim, [1.0; 4.0]);


plot(samples[2001:end, 1], samples[2001:end, 2], seriestype = :scatter)

ess_bulk(samples[2001:end, 1])
ess_bulk(samples[2001:end, 2])

# multiple chains
samples, d = stan(isogaussian, N, ndim, 4);

plot(1:2000, samples[2001:end, 2, :])

plot(samples[2001:end, 1, :], samples[2001:end, 2, :],
     seriestype = :scatter, alpha = 0.1)


round.(mean(samples[Int(N / 2):end, :, :], dims=[1,3]), digits=1)
round.(std(samples[Int(N / 2):end, :, :], dims=[1,3]), digits=1)

map(i -> ess_bulk(samples[Int(N / 2):end, i, :]), 1:ndim)
map(i -> ess_tail(samples[Int(N / 2):end, i, :]), 1:ndim)
map(i -> ess_std(samples[Int(N / 2):end, i, :]), 1:ndim)
map(i -> rhat(samples[Int(N / 2):end, i, :]), 1:ndim)

mean(d[:leapfrog][2001:end, :], dims=1)
mean(d[:acceptstat][2001:end, :], dims=1)
mean(d[:stepsize][2001:end, :], dims=1)
mean(d[:treedepth][2001:end, :], dims=1)

plot(1:2000, d[:acceptstat][2001:end, 2])

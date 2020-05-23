using Revise
using Plots; plotly();
using Statistics
using StatsPlots
using LinearAlgebra
using SparseArrays
using Distributions
using Random
using RandomNumbers.PCG
using DelimitedFiles
using BenchmarkTools
using Zygote

using Distributed; addprocs(3);
# @everywhere include("hmc.jl")
includet("src/hmc.jl")
includet("src/convergence.jl")
# @everywhere include("models.jl")
includet("src/models.jl")
includet("src/utilities.jl")


q = Dict(:xraw => 0.5)
q = prepareq(q)
d = Dict(:mu => 0.0, :sigma => 1.0)

function model_(q, d)
    d[:x] = d[:mu] + q[:xraw] * d[:sigma] # transformed parameter
    return normal_(q[:xraw], 0.0, 1.0)
end

g(q) = model_(q, d)
gg = q -> first(gradient(g, q))

gg(q)


# single chain
samples, diagnostics = hmc(g, q;
                 # M = cholesky(Symmetric(Minv)).L,
                 # M = Minv,
                 control = Dict(:skewsymmetric => false, :iterations => 2000));

samples2, d2 = hmc(f, ndim;
                 # M = cholesky(Symmetric(Minv)).L,
                 M = Minv,
                 control = Dict(:skewsymmetric => false, :iterations => 2000));



# multiple chains
p# @benchmark
samples, d = stan(f, ndim;
                  M = Minv, control = Dict(:skewsymmetric => false, :iterations => 2000));


round.(mean(samples[:, :, :], dims=1), digits=1)
round.(std(samples[:, :, :], dims=1), digits=1)

round.(mean(samples2[:, :, :], dims=1), digits=1)
round.(std(samples2[:, :, :], dims=1), digits=1)

map(i -> round(ess_bulk(samples[:, :, 1])), 1)
map(i -> round(ess_tail(samples[:, :, 1])), 1:ndim)

map(i -> round(ess_std(samples[:, i, 1]), digits=2), 1:ndim)
map(i -> round(rhat(samples[:, i, 1]), digits=2), 1:ndim)

plot(1:10000, samples[:, 2, 2])


sum(d[:divergent], dims=1)
plot(d[:leapfrog][idx, :], seriestype=:density, title="Number leapfrog steps")
plot(d[:acceptstat][idx, :], seriestype=:density, title="Accept stats")
plot(1:15000, log10.(d[:stepsize][:, 2]), seriestype=:scatter, title="Stepsizes")
plot(d[:treedepth][2001:end, :], seriestype=:density, title="Tree depths")
plot(d[:massmatrix]', seriestype=:density, title="Mass matrix")


# TODO a future test




samplevariance(w, Dict(:skewsymmetric => false, :regularize => false))

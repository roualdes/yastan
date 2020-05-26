using Revise
using YaStan
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
using SharedArrays
using Distributed; addprocs(3);

# @everywhere include("hmc.jl")
# includet("src/hmc.jl")
# includet("src/convergence.jl")
# includet("src/models.jl")
# includet("src/utilities.jl")
# @everywhere include("models.jl")



q = Dict{Symbol, Any}(:xraw => 0.5);
d = Dict{Symbol, Any}(:mu => 14.5, :sigma => 3.14);

function model_(q, d)
    # TODO figure out how to store this
    d[:x] = d[:mu] .+ q[:xraw] * d[:sigma] # transformed parameter
    return sum(YaStan.normal_.(q[:xraw], 0.0, 1.0))
end

# single chain
samples, diagnostics = YaStan.hmc(model_, q, d);


# multiple chains
# @benchmark
samples, diagnostics = stan(g, q);


round.(mean(samples, dims=1), digits=1)
round.(std(samples, dims=1), digits=1)

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

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
using SharedArrays

using Distributed; addprocs(3);
# @everywhere include("hmc.jl")
includet("src/hmc.jl")
includet("src/convergence.jl")
# @everywhere include("models.jl")
includet("src/models.jl")
includet("src/utilities.jl")


q = Dict(:xraw => 0.5 * ones(5))
q = prepareq(q)
d = convert(Dict{Any,Any}, Dict(:mu => 14.5, :sigma => 3.14))

function model_(q, d)
    # TODO figure out how to store this
    d[:x] = d[:mu] .+ q[:xraw] * d[:sigma] # transformed parameter
    return sum(normal_.(q[:xraw], 0.0, 1.0))
end

g(q) = model_(q, d)

q = Dict(:x => randn(64))
q = prepareq(q)
g(q) = normal_(q[:x], zeros(64), Matrix(Diagonal(ones(64))))

gg = q -> first(gradient(g, q))

g(q)

gg(q)


# single chain
samples, diagnostics = hmc(g, q; control = Dict(:iterations => 100_000));


# multiple chains
# @benchmark
samples, diagnostics = stan(g, q);


round.(mean(samples[:, 1:5] * d[:sigma] .+ d[:mu], dims=1), digits=2)
round.(std(samples[:, 1:5] * d[:sigma] .+ d[:mu], dims=1), digits=2)

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

using Revise
using YaStan

using Plots; plotly();
using Statistics
using StatsPlots
using LinearAlgebra
using SparseArrays
using Distributions
using BenchmarkTools
using Zygote


# @everywhere include("hmc.jl")
# includet("src/hmc.jl")
# includet("src/convergence.jl")
# includet("src/models.jl")
# includet("src/utilities.jl")
# @everywhere include("models.jl")

q = Dict{Symbol, Any}(:xraw => 1.0);
d = Dict{Symbol, Any}(:mu => 14.5, :sigma => 3.14);

function model(q, d)
    # TODO figure out how to store this
    d[:x] = d[:mu] .+ q[:xraw] * d[:sigma] # transformed parameter
    return YaStan.normal_.(q[:xraw], 0.0, 1.0)
end




N = 1000
X = randn(N);
a = 4.0
b = -2.0
Y = a .+ X * b .+ randn(N);

d = Dict{Symbol, Any}(:x => X, :y => Y);
q = Dict{Symbol, Any}(:alpha => randn(), # all Real valued parameters
                      :beta => randn(),
                      :sigma => randn());

function lm(q, d)
    L = 0.0
    prior = 0.0

    # transformed parameter
    d[:s] = exp(q[:sigma])

    mu = q[:alpha] .+ d[:x] * q[:beta]
    L += sum(YaStan.normal_.(d[:y],  mu, d[:s])) - q[:sigma]

    prior += YaStan.normal_(q[:alpha], 0.0, 3.0)
    prior += sum(YaStan.normal_.(q[:beta], 0.0, 3.0))

    prior += YaStan.exponential_(d[:s], 1 / 3.0)

    return L + prior
end

# single chain
samples, i = YaStan.hmc(lm, q, d);

mean(log.(samples[:, 3]))
round.(mean(samples, dims=1), digits=1)

round.(std(samples, dims=1), digits=1)

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

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

# using Distributed; addprocs(3);
# @everywhere include("hmc.jl")
includet("hmc.jl")
includet("convergence.jl")
# @everywhere include("models.jl")
includet("models.jl")



function funnel_(q, d)
    Ny = normal_(d[:y], θ[:m], exp(-3 * θ[:s]))
    Nm = normal_(θ[:m], 0, 1)
    Ns = normal_(θ[:s], 0, 1)
    return Ny + Nm + Ns
end

d = Dict(:y => 0.5)

f(q) = funnel_(q, )

U = Uniform(-2, 2)

for (k, v) in d
    println(rand(U, size(v)))
end




# single chain
samples, d = hmc(f, ndim;
                 # M = cholesky(Symmetric(Minv)).L,
                 # M = Minv,
                 control = Dict(:skewsymmetric => false, :iterations => 2000));

samples2, d2 = hmc(f, ndim;
                 # M = cholesky(Symmetric(Minv)).L,
                 M = Minv,
                 control = Dict(:skewsymmetric => false, :iterations => 2000));



# multiple chains
# @benchmark
samples, d = stan(f, ndim;
                  M = Minv, control = Dict(:skewsymmetric => false, :iterations => 2000));


round.(mean(samples[:, :, :], dims=1), digits=1)
round.(std(samples[:, :, :], dims=1), digits=1)

round.(mean(samples2[:, :, :], dims=1), digits=1)
round.(std(samples2[:, :, :], dims=1), digits=1)

map(i -> round(ess_bulk(samples[:, :, 1])), 1:ndim)
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
μ = zeros(2)
S = [1.0 0.99; 0.99 1.0];
N = MvNormal(μ, S)
w = WelfordState(zero(μ), zeros(length(μ)), 0)
W = WelfordState(zero(μ), zero(S), 0)
X = zeros(2, 10)

for n in 1:10
    x = rand(N)
    global X[:, n] = x
    global w = accmoments(w, x)
    global W = accmoments(W, x)
end

cov(X', corrected = true)
samplevariance(W, Dict(:skewsymmetric => true, :regularize => false))
samplevariance(w, Dict(:skewsymmetric => false, :regularize => false))

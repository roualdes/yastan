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


ndim = 64
Minv = Matrix(Diagonal(ones(ndim))); # ones(ndim);

# set up model
M = sparse([1.0 0.75; 0.75 1.0]);
S = Matrix(blockdiag([M for i in 1:ndim/2]...));
μ = zeros(ndim);

f = mvgaussian_(μ, Matrix(S));

normal_.(randn(10), 0, 10)

function Normal_(x::Vector{Float64}, μ::Vector{Float64}, Σ::Matrix{Float64})
    d = x - μ
    return 0.5 * (d' * (Σ \ d))
end

f(x) = Normal_(x, μ, S)

# single chain
samples, d = hmc(f, ndim;
                 M = cholesky(Symmetric(Minv)).L,
                 # M = Minv,
                 control = Dict(:skewsymmetric => true, :iterations => 2000));

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

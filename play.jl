using Revise
using YaStan
using Statistics
using Zygote

using Plots; plotly();
using StatsPlots
using LinearAlgebra
using SparseArrays
using Distributions
using BenchmarkTools

q = Dict{Symbol, Any}(:xraw => 1.0);
d = Dict{Symbol, Any}(:mu => 14.5, :sigma => 3.14);

function model(q, d)
    # TODO figure out how to store this
    d[:x] = d[:mu] .+ q[:xraw] * d[:sigma] # transformed parameter
    return YaStan.normal_.(q[:xraw], 0.0, 1.0)
end

function f(x)
    y = Zygote.dropgrad(x * x * x)
    println(y)
    return x * x
end

Zygote.gradient(f, 3.0)


N = 1000
X = randn(N, 3);
a = 4.0
b = [-2.0; 5.0; 1.0]
Y = a .+ X * b .+ randn(N);

d = Dict{Symbol, Any}(:x => X, :y => Y); # all real valued parameters
q = Dict{Symbol, Any}(:alpha => randn(),
                      :beta => randn(3),
                      :s => randn());

function lm(q, d)
    L = 0.0
    prior = 0.0

    mu = q[:alpha] .+ d[:x] * q[:beta]
    sigma = exp(q[:s])
    L += sum(YaStan.normal_.(d[:y],  mu, sigma)) - q[:s]

    prior += YaStan.normal_(q[:alpha], 0.0, 3.0)
    prior += sum(YaStan.normal_.(q[:beta], 0.0, 3.0))

    prior += YaStan.exponential_(sigma, 1 / 3.0) - q[:s]

    return L + prior
end

# single chain
samples, i = YaStan.hmc(lm, q, d; control = Dict(:iterations => 4000));

mean(log.(samples[:, 3]))
round.(mean(samples, dims=1), digits=1)

round.(std(samples, dims=1), digits=1)

map(i -> round(YaStan.ess_bulk(samples[:, i])), 1:4)
map(i -> round(YaStan.ess_tail(samples[:, i])), 1:4)

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

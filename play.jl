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

# using Distributed; addprocs(3);
# @everywhere include("hmc.jl")
includet("hmc.jl")
includet("convergence.jl")
# @everywhere include("models.jl")
includet("models.jl")



function funnel_(q, d)
    Ny = normal_(d[:y], q[:m], exp(-3 * q[:s]))
    Nm = normal_(q[:m], 0, 1)
    Ns = normal_(q[:s], 0, 1)
    return Ny + Nm + Ns
end
f(q) = funnel_(q, d)

U = q -> first(gradient(f, q))

U(q)



function model_(q, d)
    return normal_(d[:y], q[:m], q[:s])
end
g(q) = model_(q, d)

d = Dict(:y => 0.5)

gg = q -> first(gradient(g, q))
gg(q)


U = Uniform(-2, 2)
q = Dict(:m => rand(U), :s => rand(U, (2, 2)), :b => rand(U, 1))

function prepareq!(q::Dict)
    q[:vec] = Dict()
    l = zeros(Int, length(q))
    idx = 1
    ndims = 0
    for (i, (k, v)) in enumerate(q)
        if k == :vec
            continue
        end
        l[i] = length(v)
        jdx = idx + l[i] - 1
        if idx == jdx
            q[:vec][k] = idx
        else
            q[:vec][k] = idx:jdx
        end
        ndims += length(q[:vec][k])
        idx += l[i]
    end
    q[:ndims] = ndims
    return
end

v = [1.0; 2.0; 3.0; 4.0; 0.5; 0.6]

function updateq!(q::Dict, v::Vector{Float64}; addself = false)
    @assert length(v) == q[:ndims] "Can't assign q into vector v, differing lengths"
    for k in keys(q[:vec])
        idx = q[:vec][k]
        l = length(idx)
        if l > 1
            tmpv = reshape(v[idx], size(q[k]))
            q[k] .= addself ? q[k] + tmpv : tmpv
        else
            q[k] = addself ? q[k] + v[idx] : v[idx]
        end
    end
    return
end


function assignq!(v::Vector{Float64}, q::Dict)
    @assert length(v) == q[:ndims] "Can't assign q into vector v, differing lengths"
    for k in keys(q[:vec])
        idx = q[:vec][k]
        if length(idx) > 1
            v[idx] .= vec(q[k])
        else
            v[idx] = q[k]
        end
    end
    return
end


function vecd(d::Dict)
    l = zeros(Int, length(d))
    for (i, v) in enumerate(values(d))
        l[i] = length(v)
    end

    v = zeros(sum(l))
    idx = 1
    for (i, val) in enumerate(values(d))
        jdx = idx + l[i] - 1
        if idx == jdx
            v[idx:jdx] .= val
        else
            v[idx:jdx] .= vec(val)
        end
        idx += l[i]
    end
    return v
end



function dictv!(q::Vector{Float64}, d::Dict)
    # TODO a view on q? would minimize copying
    idx = 1
    for (k, v) in d
        l = length(v)
        jdx = idx + l - 1
        if idx == jdx
            d[k] = q[idx:jdx][1]
        else
            d[k] = reshape(q[idx:jdx], size(v))
        end
        idx += l
    end
    return d
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

using YaStan
using Test
using LinearAlgebra
using Distributions
using Statistics

function test_welford()
    μ = zeros(2)
    S = [1.0 0.99; 0.99 1.0];
    N = MvNormal(μ, S)
    w = YaStan.WelfordState(zero(μ), zeros(length(μ)), 0)
    W = YaStan.WelfordState(zero(μ), zero(S), 0)
    X = zeros(2, 10)

    for n in 1:10
        x = rand(N)
        X[:, n] = x
        w = YaStan.accmoments(w, x)
        W = YaStan.accmoments(W, x)
    end

    Welfordestimate = YaStan.samplevariance(W, Dict(:skewsymmetric => false, :regularize => false))
    welfordestimate = YaStan.samplevariance(w, Dict(:skewsymmetric => false, :regularize => false))

    var = cov(X', corrected = true)

    return isapprox(Welfordestimate, var) && isapprox(welfordestimate, diag(var))
end


@test test_welford() == true

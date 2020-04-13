using LinearAlgebra

# TODO (ear) solve stupid 1d things going on
function gaussian_(μ, σ)
    c = 0.5 / (σ * σ)
    return function(x)
        d = x .- μ
        return c * d .* d
    end
end

function t_(nu::Float64, s::Float64)
    return function(x)
        n = 0.5 * x[1] * x[1]
        ig = 0.5 * nu * x[2] + 0.5 * nu * s * s * exp(-x[2])
        return n + ig
    end
end

# correlated 2-d Gaussian
function anisotropicgaussian(θ)
    A = [50.251256 -24.874372;
         -24.874372 12.562814]
    # Σ = inv(A)
    return 0.5 * θ' * A * θ
end

A = [1.0 1.98; 1.98 4.0]

function mvgaussian_(μ::AbstractVector{Float64},
                     Σ::AbstractMatrix{Float64})
    return function(x)
        M = Σ \ (x - μ)
        return 0.5 * (x - μ)' * M
    end
end





## below needs lots of reviewing
# logistic
function logistic(x)
    return 1 / (1 + exp(-x))
end

function genlogistic(N = 1001, K = 3, a = Normal(0, 10), b = Normal(0, 2.5), x = Normal(0, 10))
    α = rand(a)
    β = rand(b, K)
    θ = [α; β]
    X = [ones(N) rand(x, (N, K))]
    μ = X * θ
    p = logistic.(μ)
    y = convert(Vector{Int}, rand.(Bernoulli.(p)))
    return y, X, θ
end

# df = DataFrame(y=Y, x2 = X[:, 2], x3 = X[:, 3], x4 = X[:, 4])
# fit = glm(@formula(y ~ x2 + x3 + x4), df, Bernoulli())

function logisticmodel(y, X, θ)
    μ = X * θ
    β = θ[2:end-1]
    return -y' * μ + sum(log1p.(exp.(μ))) + 0.5 + θ[1] * θ[1] / 100 + 0.5 * β' * β / 10
end
# logisticreg(θ) = logisticmodel(y, X, θ)


function hierlogisticmodel(y, X, θ)
    β = θ[1:(end - 1)]
    K = length(β)
    μ = X * β
    likelihood =
    σ = θ[end]
    κ = exp(σ)
    prior = -0.5 * β' * β / κ ^ 2 - K * σ - 0.1 * κ + σ
    return likelihood + prior
end

heirlogisticreg(θ) = hierlogisticmodel(θ, y, X)

# linear
function genlinear(N = 1001, K = 3, a = Normal(0, 10), b = Normal(0, 2.5), x = Normal(0, 10))
    α = rand(a)
    β = rand(b, K)
    θ = [α; β]
    X = [ones(N) rand(x, (N, K))]
    μ = X * θ
    y = rand.(Normal.(μ, 1))
    return y, X, θ
end

# df = DataFrame(y=Y, x2 = X[:, 2], x3 = X[:, 3], x4 = X[:, 4])
# fit = glm(@formula(y ~ x2 + x3 + x4), df, Normal())

function linearmodel(y, X, θ)
    e = y - X * θ
    return 0.5 * e' * e + 0.5 * θ' * θ / 25
end
# linearreg(θ) = linearmodel(y, X, θ)

function hierlinearmodel(y, X, θ)
    β = θ[1:(end - 1)]
    K = length(β)
    σ = θ[end]
    e = y - X * β
    return 0.5 * e' * e + K*σ + 0.5 * θ' * θ / exp(2 * σ) + 0.5 * exp(2 * σ) / 25 - σ
end

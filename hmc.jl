using LinearAlgebra
using Distributions
using ForwardDiff
using Random
using RandomNumbers.PCG
using Distributed
@everywhere using SharedArrays

struct PSPoint
    q::Vector{Float64}
    p::Vector{Float64}
end


struct LikelihoodGrad
    l
    grad
end

function maybeupdate!(c::Dict, s::Symbol, v)
    if !haskey(c, s)
        merge!(c, Dict(s => v))
    end
    return nothing
end

function checkcontrol(c::Dict)

    c = convert(Dict{Any, Any}, c)

    # TODO (ear) add checks on all values
    maybeupdate!(c, :iterations, 2000)
    maybeupdate!(c, :iterations_warmup, c[:iterations] ÷ 2)
    maybeupdate!(c, :chains, 4)
    maybeupdate!(c, :chainid, 1)
    maybeupdate!(c, :rng, PCGStateOneseq(UInt64, PCG_RXS_M_XS))
    maybeupdate!(c, :maxtreedepth, 10)
    maybeupdate!(c, :μ, log(10))
    maybeupdate!(c, :adaptinitpercent, 0.15)
    maybeupdate!(c, :adapttermpercent, 0.1)
    maybeupdate!(c, :adaptwindowsize, 25)
    maybeupdate!(c, :initradius, 2)
    maybeupdate!(c, :initattempts, 100)
    maybeupdate!(c, :δ, 0.8)
    maybeupdate!(c, :γ, 0.05)
    maybeupdate!(c, :t0, 10)
    maybeupdate!(c, :κ, 0.75)
    maybeupdate!(c, :regularize, true)

    return c
end

function stan(ℓ, ndim;
              M::Array{Float64} = ones(ndim),
              control::Dict = Dict())

    control = checkcontrol(control)
    I = control[:iterations] + control[:iterations_warmup]
    chains = control[:chains]

    samples = SharedArray{Float64}(control[:iterations], chains, ndim)
    leaps = SharedArray{Float64}(I, chains)
    acceptstats = SharedArray{Float64}(I, chains)
    stepsizes = SharedArray{Float64}(I, chains)
    treedepths = SharedArray{Float64}(I, chains)
    energies = SharedArray{Float64}(I, chains)
    divergences = SharedArray{Bool}(I, chains)
    massmatrices = SharedArray{Float64}(ndim, chains)

    @sync @distributed for chain in 1:control[:chains]
        control = merge(control, Dict(:chainid => chain))
        samples[:, chain, :], d = hmc(ℓ, ndim; M = M, control = control)
        leaps[:, chain] = d[:leapfrog]
        acceptstats[:, chain] = d[:acceptstat]
        stepsizes[:, chain] = d[:stepsize]
        treedepths[:, chain] = d[:treedepth]
        energies[:, chain] = d[:energy]
        divergences[:, chain] = d[:divergent]
        massmatrices[:, chain] = d[:massmatrix]
    end
    return samples, Dict(:leapfrog => leaps,
                         :acceptstat => acceptstats,
                         :stepsize => stepsizes,
                         :treedepth => treedepths,
                         :energy => energies,
                         :divergent => divergences,
                         :massmatrix => massmatrices,
                         :control => control)
end

function hmc(ℓ, ndim;
             M::Array{Float64} = ones(ndim),
             control::Dict = Dict())

    control = checkcontrol(control)
    I = control[:iterations] + control[:iterations_warmup]

    # advance RandomNumbers RNG
    # TODO (ear) figure out their type structure
    # and document this requirement.
    advance!(control[:rng], convert(UInt64, 1 << 50 * control[:chainid]))

    L = LikelihoodGrad(ℓ, q -> ForwardDiff.gradient(ℓ, q))
    q = initializesample(ndim, L, control)

    samples = zeros(I, ndim)
    samples[1, :] = q

    N = Normal()
    U = Uniform()
    zsample = PSPoint(q, generatemomentum(control[:rng], ndim, M))

    ε = findepsilon(1.0, zsample, L, M, control)
    μ = control[:μ]
    εbar = 0.0
    sbar = 0.0
    xbar = 0.0

    W = WelfordStates(zeros(ndim), zero(M), 0)
    openwindow, closewindow, lastwindow, windowstep =
        adaptionwindows(control)

    leaps = zeros(Int, I)
    treedepths = zeros(Int, I)
    energies = zeros(I)
    acceptstats = zeros(I)
    εs = zeros(I)
    divergences = falses(I)
    stepsizecounter = 0

    for i = 2:I
        z = PSPoint(samples[i - 1, :], generatemomentum(control[:rng], ndim, M))
        H0 = hamiltonian(L, z, M)
        H = H0

        zf = z
        zb = z
        zsample = z
        zpr = z

        # Momentum and sharp momentum at forward end of forward subtree
        pff = z.p
        psharpff = rhosharp(z.p, M) # dtau_dp

        # Momentum and sharp momentum at backward end of forward subtree
        pfb = z.p
        psharpfb = psharpff

        # Momentum and sharp momentum at forward end of backward subtree
        pbf = z.p
        psharpbf = psharpff

        # Momentum and sharp momentum at backward end of backward subtree
        pbb = z.p
        psharpbb = psharpff

        # Integrated momenta along trajectory
        rho = z.p

        α = 0.0
        depth = 0
        nleapfrog = 0
        lsw = 0.0

        while depth < control[:maxtreedepth]

            rhof = zeros(length(rho))
            rhob = zeros(length(rho))
            lswsubtree = -Inf

            if rand(control[:rng]) > 0.5
                rhob = rho
                pbf = pff
                psharpbf = psharpff

                zf, zpr, validsubtree,
                psharpfb, psharpff, rhof, pfb, pff,
                nleapfrog, lswsubtree, α  =
                    buildtree(depth, zf,
                              psharpfb, psharpff, rhof, pfb, pff,
                              H0, 1 * ε, L, M,
                              nleapfrog, lswsubtree, α, control[:rng])
            else
                rhof = rho
                pfb = pbb
                psharpfb = psharpbb

                zb, zpr, validsubtree,
                psharpbf, psharpbb, rhob, pbf, pbb,
                nleapfrog, lswsubtree, α =
                    buildtree(depth, zb,
                              psharpbf, psharpbb, rhob, pbf, pbb,
                              H0, -1 * ε, L, M,
                              nleapfrog, lswsubtree, α, control[:rng])
            end

            if !validsubtree
                divergences[i] = true
                break
            end
            depth += 1

            if lswsubtree > lsw
                zsample = zpr
            else
                if rand(control[:rng]) < exp(lswsubtree - lsw)
                    zsample = zpr
                end
            end

            lsw = logsumexp(lsw, lswsubtree)

            # Demand satisfication around merged subtrees
            rho = rhob + rhof
            persist = stancriterion(psharpbb, psharpff, rho)

            # Demand satisfaction between subtrees
            rhoextended = rhob + pfb
            persist &= stancriterion(psharpbb, psharpfb, rhoextended)

            rhoextended = rhof + pbf
            persist &= stancriterion(psharpbf, psharpff, rhoextended)

            if !persist
                break
            end
        end # end while

        samples[i, :] = zsample.q
        energies[i] = hamiltonian(L, zsample, M)
        treedepths[i] = depth
        leaps[i] = nleapfrog
        εs[i] = ε
        acceptstats[i] = α / nleapfrog

        if i < control[:iterations_warmup]
            ε, εbar, stepsizecounter, sbar, xbar =
                adaptstepsize(acceptstats[i], stepsizecounter, sbar, xbar, μ, control)

            if openwindow <= i <= lastwindow
                W = accmoments(W, samples[i, :])
            end

            if i == closewindow
                # reset var
                M = samplevariance(W, control)
                W = WelfordStates(zeros(ndim), zero(M), 0)

                # reset stepsize
                ε = findepsilon(ε, zsample, L, M, control)
                μ = log(10 * ε)
                sbar = 0.0
                xbar = 0.0
                stepsizecounter = 0

                # update windows
                windowstep *= 2
                closewindow += windowstep
                nextclosewindow = closewindow + windowstep * 2
                if nextclosewindow > lastwindow
                    closewindow = lastwindow
                end
            end
        else
            ε = εbar
        end

    end # end for

    postwarmup = control[:iterations_warmup]+1:I
    return samples[postwarmup, :, :], Dict(:stepsize => εs,
                                           :leapfrog => leaps,
                                           :acceptstat => acceptstats,
                                           :treedepth => treedepths,
                                           :energy => energies,
                                           :divergent => divergences,
                                           :massmatrix => M)
end


function buildtree(depth::Int, z::PSPoint,
                    psharpbeg, psharpend, rho, pbeg, pend,
                    H0::Float64, ε::Float64, L, M::Array{Float64},
                    nleapfrog::Int, logsumweight::Float64, α::Float64, rng)
    if depth == 0
        zpr = leapfrog(z, L, ε, M)
        z = zpr
        nleapfrog += 1

        H = hamiltonian(L, zpr, M)
        if isnan(H)
            H = Inf
        end

        divergent = false
        if H - H0 > 1000.0
            divergent = true
        end

        Δ = H0 - H
        logsumweight = logsumexp(logsumweight, Δ)
        α += Δ > 0.0 ? 1.0 : exp(Δ)

        psharpbeg = rhosharp(zpr.p, M) # dtau_dp
        psharpend = psharpbeg

        rho += zpr.p
        pbeg = zpr.p
        pend = pbeg

        return z, zpr, !divergent, psharpbeg, psharpend, rho, pbeg, pend, nleapfrog, logsumweight, α
    end

    lswinit = -Inf

    psharpinitend = similar(z.p)
    rhoinit = zeros(length(rho))
    pinitend = similar(z.p)

    z, zpr, validinit,
    psharpbeg, psharpinitend, rhoinit, pbeg, pinitend,
    nleapfrog, lswinit, α =
        buildtree(depth - 1, z,
                   psharpbeg, psharpinitend, rhoinit, pbeg, pinitend,
                   H0, ε, L, M, nleapfrog, lswinit, α, rng)

    if !validinit
        return zpr, zpr, false,
        psharpbeg, psharpend, rho, pbeg, pend,
        nleapfrog, logsumweight, α
    end

    lswfinal = -Inf

    psharpfinalbeg = similar(z.p)
    rhofinal = zeros(length(rho))
    pfinalbeg = similar(z.p)

    z, zfinalpr, validfinal,
    psharpfinalbeg, psharpend, rhofinal, pfinalbeg, pend,
    nleapfrog, lswfinal, α =
        buildtree(depth - 1, z,
                   psharpfinalbeg, psharpend, rhofinal, pfinalbeg, pend,
                   H0, ε, L, M, nleapfrog, lswfinal, α, rng)

    if !validfinal
        return zfinalpr, zfinalpr, false,
        psharpbeg, psharpend, rho, pbeg, pend,
        nleapfrog, logsumweight, α
    end

    lswsubtree = logsumexp(lswinit, lswfinal)

    if lswfinal > lswsubtree
        zpr = zfinalpr
    else
        if rand(rng) < exp(lswfinal - lswsubtree)
            zpr = zfinalpr
        end
    end

    logsumweight = logsumexp(logsumweight, lswsubtree)

    rhosubtree = rhoinit + rhofinal
    rho += rhosubtree

    # Demand satisfaction around merged subtrees
    persist = stancriterion(psharpbeg, psharpend, rhosubtree)

    # Demand satisfaction between subtrees
    rhosubtree = rhoinit + pfinalbeg
    persist &= stancriterion(psharpbeg, psharpfinalbeg, rhosubtree)

    rhosubtree = rhofinal + pinitend
    persist &= stancriterion(psharpinitend, psharpend, rhosubtree)

    return z, zpr, persist,
    psharpbeg, psharpend, rho, pbeg, pend,
    nleapfrog, logsumweight, α
end


function hamiltonian(L::LikelihoodGrad, z::PSPoint, M::Vector{Float64})::Float64
    return L.l(z.q) + 0.5 * rhosharp(z.p, M)' * z.p
end

function hamiltonian(L::LikelihoodGrad, z::PSPoint, M::Matrix{Float64})::Float64
    return L.l(z.q) + 0.5 * z.p' * M * z.p
end


function leapfrog(z::PSPoint, L::LikelihoodGrad,
                  ε::Float64, M::Array{Float64})::PSPoint
    p_ = z.p - 0.5 * ε * L.grad(z.q)
    q = z.q + ε * rhosharp(p_, M)
    p = p_ - 0.5 * ε * L.grad(q)
    return PSPoint(q, p)
end


function rhosharp(p::Vector{Float64}, M::Vector{Float64})
    return p .* M
end

function rhosharp(p::Vector{Float64}, M::Matrix{Float64})
    return M * p
end


function stancriterion(psharp_m::Vector{Float64}, psharp_p::Vector{Float64},
                       rho::Vector{Float64})::Bool
    return psharp_p' * rho > 0 && psharp_m' * rho > 0
end


function generatemomentum(rng, ndim::Int, M::Vector{Float64})::Vector{Float64}
    return randn(rng, ndim) ./ sqrt.(M)
end

function generatemomentum(rng, ndim::Int, M::Matrix{Float64})::Vector{Float64}
    return cholesky(Hermitian(M)).U \ randn(rng, ndim)
end


function findepsilon(ε::Float64, z::PSPoint, L::LikelihoodGrad,
    M::Array{Float64}, control)::Float64
    ndim = length(z.q)
    H0 = hamiltonian(L, z, M)

    zp = leapfrog(z, L, ε, M)
    H = hamiltonian(L, zp, M)
    isnan(H) && (H = Inf)

    ΔH = H0 - H
    direction = ΔH > log(0.8) ? 1 : -1

    while true
        rp = generatemomentum(control[:rng], ndim, M)
        H0 = hamiltonian(L, z, M)

        zp = leapfrog(z, L, ε, M)
        H = hamiltonian(L, zp, M)
        isnan(H) && (H = Inf)

        ΔH = H0 - H
        if direction == 1 && !(ΔH > log(0.8))
            break
        elseif direction == -1 && !(ΔH < log(0.8))
            break
        else
            ε = direction == 1 ? 2.0 * ε : 0.5 * ε
        end

        @assert ε <= 1.0e7 "Posterior is impropoer.  Please check your model."
        @assert ε >= 0.0 "No acceptable small step size could be found.  Perhaps the posterior is not continuous."
    end

    return ε
end


function adaptionwindows(control)

    # TODO(ear) check, Madapt * termpercent > 20
    openwindow::Int = floor(control[:iterations_warmup] *
        control[:adaptinitpercent])

    lastwindow::Int = ceil(control[:iterations_warmup] *
        (1 - control[:adapttermpercent]))

    return openwindow, openwindow + control[:adaptwindowsize],
    lastwindow, control[:adaptwindowsize]
end


function initializesample(ndim::Int, L::LikelihoodGrad, control)::Vector{Float64}
    q = zeros(ndim)
    initialized = false
    a = 0

    while a < control[:initattempts] && !initialized
        q = generateuniform(control[:rng], length(q), control[:initradius])

        lq = L.l(q)
        if isfinite(lq) && !isnan(lq)
            initialized = true
        end

        gq = sum(L.grad(q))
        if isfinite(gq) && !isnan(gq)
            initialized &= true
        end

        a += 1
    end

    @assert a <= control[:initattempts] && initialized "Failed to find inital values in $(initattempts) attempts."

    return q
end


function generateuniform(rng, N, radius)
    return rand(rng, N) .* (radius * 2) .- radius
end


function adaptstepsize(adaptstat, counter, sbar, xbar, μ, control)
    counter += 1
    adaptstat = adaptstat > 1 ? 1 : adaptstat
    eta = 1.0 / (counter + control[:t0])
    sbar = (1.0 - eta) * sbar + eta * (control[:δ] - adaptstat)
    x = μ - sbar * sqrt(counter) / control[:γ]
    xeta = counter ^ -control[:κ]
    xbar = (1.0 -  xeta) * xbar + xeta * x
    ε = exp(x)
    εbar = exp(xbar)
    return ε, εbar, counter, sbar, xbar
end


function log1pexp(a)
    if a > 0.0
        return a + log1p(exp(-a))
    end
    return log1p(exp(a))
end


function logsumexp(a, b)
    if a == -Inf
        return b
    end

    if a == Inf && b == Inf
        return Inf
    end

    if a > b
        return a + log1pexp(b - a)
    end

    return b + log1pexp(a - b)
end


struct WelfordStates{S<:Vector{Float64}, T<:Array{Float64}}
    m::S
    s::T
    n::Int
end


function accmoments(ws::WelfordStates{Vector{Float64}, Vector{Float64}},
                    x::Vector{Float64})::WelfordStates
    n = ws.n + 1
    d = similar(ws.m)
    M = similar(ws.m)
    S = similar(ws.s)
    @. d = x - ws.m
    @. M = ws.m + d / n
    @. S = ws.s + d * (x - M)
    return WelfordStates(M, S, n)
end

function accmoments(ws::WelfordStates{Vector{Float64}, Matrix{Float64}},
                    x::Vector{Float64})::WelfordStates
    n = ws.n + 1
    d = similar(ws.m)
    M = similar(ws.m)
    S = similar(ws.s)
    @. d = x - ws.m
    @. M = ws.m + d / n
    S = ws.s .+ (x .- M) * d'
    return WelfordStates(M, S, n)
end


function samplevariance(ws::WelfordStates{Vector{Float64}, Vector{Float64}},
                        control)::Vector{Float64}
    if ws.n > 1
        σ = similar(ws.s)
        @. σ = ws.s / (ws.n - 1)
        if control[:regularize]
            w = ws.n / (ws.n + 5.0)
            σ = w * σ + 1e-3 * (1 - w) * ones(length(σ))
        end
        return σ
    end
    return ones(length(ws.m))
end

function samplevariance(ws::WelfordStates{Vector{Float64}, Matrix{Float64}},
                        control)::Matrix{Float64}
    if ws.n > 1
        σ = similar(ws.s)
        @. σ = ws.s / (ws.n - 1)
        if control[:regularize]
            w = ws.n / (ws.n + 5.0)
            σ = w * σ + 1e-3 * (1 - w) * one(σ)
        end
        return σ
    end
    return one(ws.s)
end

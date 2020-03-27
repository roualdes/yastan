using LinearAlgebra
using Distributions
using ForwardDiff
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


function hamiltonian(L::LikelihoodGrad, z::PSPoint, M::Vector{Float64})::Float64
    return L.l(z.q) + 0.5 * (z.p .* M)' * z.p
end


function leapfrog(z::PSPoint, L::LikelihoodGrad,
                  ε::Float64, M::Vector{Float64})::PSPoint
    p_ = z.p - 0.5 * ε * L.grad(z.q)
    q = z.q + ε * p_ .* M
    p = p_ - 0.5 * ε * L.grad(q)
    return PSPoint(q, p)
end


function stancriterion(psharp_m, psharp_p, rho)
    return psharp_p' * rho > 0 && psharp_m' * rho > 0
end


function stan(ℓ, I, ndim, chains)
    samples = SharedArray{Float64}(I, ndim, chains)
    leaps = SharedArray{Float64}(I, chains)
    acceptstats = SharedArray{Float64}(I, chains)
    stepsizes = SharedArray{Float64}(I, chains)
    depths = SharedArray{Float64}(I, chains)
    energies = SharedArray{Float64}(I, chains)
    massmatrices = SharedArray{Float64}(ndim, chains)

    @distributed for c = 1:chains
        samples[:, :, c], leaps[:, c], acceptstats[:, c],
        stepsizes[:, c], depths[:, c], energies[:, c],
        massmatrices[:, c] = hmc(ℓ, I, ndim)
    end
    return samples, Dict(:leapfrog => leaps, :acceptstat => acceptstats,
                         :stepsize => stepsizes, :treedepth => depths,
                         :energy => energies, :massmatrix => massmatrices)
end

function hmc(ℓ, I, ndim, M = ones(ndim))
    IAdapt = I ÷ 2

    L = LikelihoodGrad(ℓ, q -> ForwardDiff.gradient(ℓ, q))
    q = initializesample(ndim, L)

    samples = zeros(I, ndim)
    samples[1, :] = q

    N = Normal()
    U = Uniform()
    zsample = PSPoint(q, rand(N, ndim) ./ sqrt.(M))

    ε = findepsilon(1.0, zsample, L, M)
    εbar = 0.0
    sbar = 0.0
    μ = 2.302585092994046 # log(10)
    xbar = 0.0

    W = WelfordStates(zeros(ndim), zeros(ndim), 0)
    openwindow, closewindow, lastwindow, windowstep = adaptionwindows(IAdapt)

    leaps::Vector{Int} = zeros(I)
    maxdepth = 10
    treedepth::Vector{Int} = zeros(I)
    energy = zeros(I)
    acceptstats = zeros(I)
    εs = zeros(I)
    stepsizecounter = 0

    for i = 2:I
        z = PSPoint(samples[i - 1, :], rand(N, ndim) ./ sqrt.(M))
        H0 = hamiltonian(L, z, M)

        zf = z
        zb = zf

        zsample = zf
        zpr = zf

        # Momentum and sharp momentum at forward end of forward subtree
        pff = z.p
        psharpff = z.p .* M # dtau_dp

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

        while depth < maxdepth

            rhof = zeros(length(rho))
            rhob = zeros(length(rho))

            lswsubtree = -Inf

            if rand(U) > 0.5
                rhob = rho
                pbf = pff
                psharpbf = psharpff

                zf, validsubtree,
                psharpfb, psharpff, rhof, pfb, pff,
                nleapfrog, lswsubtree, α =
                    buildtree(depth, zf,
                               psharpfb, psharpff, rhof, pfb, pff,
                               H0, 1, ε, L, M, nleapfrog, lswsubtree, α)
                zpr = zf

            else
                rhof = rho
                pfb = pbb
                psharpfb = psharpbb

                zb, validsubtree,
                psharpbf, psharpbb, rhob, pbf, pbb,
                nleapfrog, lswsubtree, α =
                    buildtree(depth, zb,
                               psharpbf, psharpbb, rhob, pbf, pbb,
                               H0, -1, ε, L, M, nleapfrog, lswsubtree, α)
                zpr = zb

            end

            if !validsubtree
                break
            end

            depth += 1

            if lswsubtree > lsw
                zsample = zpr
            else
                if rand(U) < exp(lswsubtree - lsw)
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
        energy[i] = hamiltonian(L, zsample, M)
        treedepth[i] = depth
        leaps[i] = nleapfrog
        εs[i] = ε
        acceptstats[i] = α / nleapfrog

        if i <= IAdapt
            ε, εbar, stepsizecounter, sbar, xbar =
                adaptstepsize(acceptstats[i], stepsizecounter, sbar, xbar, μ)

            if openwindow <= i <= lastwindow
                W = accmoments(W, samples[i, :])
            end

            if i == closewindow
                # reset var
                M = samplevariance(W)
                W = WelfordStates(zeros(ndim), zeros(ndim), 0)

                # reset stepsize
                ε = findepsilon(ε, zsample, L, M)
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

    return samples, leaps, acceptstats, εs, treedepth, energy, M
end


function buildtree(depth::Int, z::PSPoint,
                    psharpbeg, psharpend, rho, pbeg, pend,
                    H0::Float64, v::Int, ε::Float64, L, M::Vector{Float64},
                    nleapfrog::Int, logsumweight::Float64, α::Float64)
    if depth == 0
        zpr = leapfrog(z, L, v * ε, M)
        nleapfrog += 1

        H = hamiltonian(L, zpr, M)
        isnan(H) && (H = Inf)

        divergent = false
        H - H0 > 1000.0 && (divergent = true)

        Δ = H0 - H
        lsw = logsumexp(logsumweight, Δ)
        α += Δ > 0.0 ? 1.0 : exp(Δ)

        psharpbeg =  zpr.p .* M # dtau_dp
        psharpend = psharpbeg

        rho += zpr.p
        pbeg = zpr.p
        pend = pbeg

        return zpr, !divergent, psharpbeg, psharpend, rho, pbeg, pend, nleapfrog, lsw, α
    end

    lswinit = -Inf

    psharpinitend = similar(z.p)
    rhoinit = zeros(length(rho))
    pinitend = similar(z.p)

    zpr, validinit,
    psharpbeg, psharpinitend, rhoinit, pbeg, pinitend,
    nleapfrog, lswinit, α =
        buildtree(depth - 1, z,
                   psharpbeg, psharpinitend, rhoinit, pbeg, pinitend,
                   H0, v, ε, L, M, nleapfrog, lswinit, α)

    if !validinit
        return z, false,
        psharpbeg, psharpend, rho, pbeg, pend,
        nleapfrog, logsumweight, α
    end

    lswfinal = -Inf

    psharpfinalbeg = similar(z.p)
    rhofinal = zeros(length(rho))
    pfinalbeg = similar(z.p)

    zfinal, validfinal,
    psharpfinalbeg, psharpend, rhofinal, pfinalbeg, pend,
    nleapfrog, lswfinal, α =
        buildtree(depth - 1, zpr,
                   psharpfinalbeg, psharpend, rhofinal, pfinalbeg, pend,
                   H0, v, ε, L, M, nleapfrog, lswfinal, α)

    if !validfinal
        return zpr, false,
        psharpbeg, psharpend, rho, pbeg, pend,
        nleapfrog, logsumweight, α
    end

    lswsubtree = logsumexp(lswinit, lswfinal)
    if lswfinal > lswsubtree
        zpr = zfinal
    else
        if rand(Uniform()) < exp(lswfinal - lswsubtree)
            zpr = zfinal
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

    return zpr, persist, psharpbeg, psharpend, rho, pbeg, pend, nleapfrog, logsumweight, α
end


function findepsilon(ε::Float64, z::PSPoint, L::LikelihoodGrad, M::Vector{Float64})::Float64
    ndim = length(z.q)
    N = Normal()
    H0 = hamiltonian(L, z, M)

    zp = leapfrog(z, L, ε, M)
    H = hamiltonian(L, zp, M)
    if isnan(H)
        H = Inf
    end

    ΔH = H0 - H
    direction = ΔH > log(0.8) ? 1 : -1

    while true
        rp = rand(N, ndim) ./ sqrt.(M)
        H0 = hamiltonian(L, z, M)

        zp = leapfrog(z, L, ε, M)
        H = hamiltonian(L, zp, M)
        if isnan(H)
            H = Inf
        end

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


function adaptionwindows(Madapt::Int; initpercent::Float64 = 0.15,
                         termpercent::Float64 = 0.1, windowstep::Int = 25)
    # TODO(ear) check, Madapt * termpercent > 20
    openwindow::Int = 75 # floor(Madapt * initpercent)
    lastwindow::Int = Madapt - 50 # ceil(Madapt * (1 - termpercent))
    return openwindow, openwindow + windowstep, lastwindow, windowstep
end


function initializesample(ndim::Int, L::LikelihoodGrad,
                          radius::Number = 2, attempts::Int = 100)::Vector{Float64}
    q = zeros(ndim)
    initialized = false
    a = 0
    U = Uniform(-radius, radius)

    while a < attempts && !initialized
        q = rand(U, ndim)

        lq = L.l(q)
        if isfinite(lq) && !isnan(lq)
            initialized = true
        end

        gq = L.grad(q)' * ones(ndim)
        if isfinite(gq) && !isnan(gq)
            initialized &= true
        end

        a += 1
    end

    @assert a <= attempts && initialized "Failed to find inital values in $(initattempts) attempts."

    return q
end


function adaptstepsize(adaptstat, counter, sbar, xbar, μ,
                       δ = 0.8, γ = 0.05, t0 = 10, κ = 0.75)
    counter += 1
    adaptstat = adaptstat > 1 ? 1 : adaptstat
    eta = 1.0 / (counter + t0)
    sbar = (1.0 - eta) * sbar + eta * (δ - adaptstat)
    x = μ - sbar * sqrt(counter) / γ
    xeta = counter ^ (-κ)
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


struct WelfordStates
    m::Vector{Float64}
    s::Vector{Float64}
    n::Int
end


function accmoments(ws::WelfordStates, x::Vector{Float64})::WelfordStates
    n = ws.n + 1
    d = similar(x)
    M = similar(x)
    S = similar(x)
    @. d = x - ws.m
    @. M = ws.m + d / n
    @. S = ws.s + d * (x - M)
    return WelfordStates(M, S, n)
end


function samplevariance(ws::WelfordStates, regularized = true)::Vector{Float64}
    if ws.n > 1
        σ = ws.s ./ (ws.n - 1)
        if regularized
            σ = (ws.n / (ws.n + 5.0)) * σ + 1.0e-3 * (5.0 / (ws.n + 5.0)) * ones(length(σ))
        end
        return σ
    end
    return ones(length(ws.m))
end

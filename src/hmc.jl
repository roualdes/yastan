function maybeupdate!(c::Dict, s::Symbol, v)
    if !haskey(c, s)
        merge!(c, Dict(s => v))
    end
    return nothing
end

function checkcontrol(c::Dict)

    # TODO would like to avoid making this copy
    # create c = Dict{Symbol, Any}(...)
    c = convert(Dict{Any, Any}, c)

    # TODO (ear) add checks on all values
    maybeupdate!(c, :iterations, 2000)
    maybeupdate!(c, :iterations_warmup, c[:iterations] ÷ 2)
    maybeupdate!(c, :chains, 4)
    maybeupdate!(c, :metric, "diag")
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
    maybeupdate!(c, :skewsymmetric, false)

    return c
end

function sample(model, q, d; control::Dict = Dict())

    control = checkcontrol(control)

    if !haskey(q, :vec)
        prepareparameters!(q, keys(q))
    end

    U(q) = model(q, d)
    ∇U = q -> first(Zygote.gradient(U, q))

    datakeys = collect(keys(d))
    checkinitialization(q, U, ∇U)
    transformedparameterskeys = setdiff(keys(d), datakeys)

    if !haskey(d, :vec)
        prepareparameters!(d, transformedparameterskeys)
    end

    # store parameters + transfomred parameters
    ndim = q[:length]
    samplesdim = ndim + d[:length]
    I = control[:iterations] + control[:iterations_warmup]

    chains = control[:chains]
    M = setmetric(control[:metric], ndim)

    samples = SharedArray{Float64}(control[:iterations], chains, ndim)
    leaps = SharedArray{Float64}(I, chains)
    acceptstats = SharedArray{Float64}(I, chains)
    stepsizes = SharedArray{Float64}(I, chains)
    treedepths = SharedArray{Float64}(I, chains)
    energies = SharedArray{Float64}(I, chains)
    divergences = SharedArray{Bool}(I, chains)
    massmatrices = SharedArray{Float64}(size(M)..., chains)

    @sync @distributed for chain in 1:control[:chains]
        control = merge(control, Dict(:chainid => chain))
        samples[:, chain, :], d = hmc(U, q, d; control = control)

        leaps[:, chain] = d[:leapfrog]
        acceptstats[:, chain] = d[:acceptstat]
        stepsizes[:, chain] = d[:stepsize]
        treedepths[:, chain] = d[:treedepth]
        energies[:, chain] = d[:energy]
        divergences[:, chain] = d[:divergent]
        # TODO (ear) there's got to be a better way
        massmatrices[fill((:), length(size(M)))..., chain] = d[:massmatrix]
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

function hmc(model, q, d; control::Dict = Dict())

    control = checkcontrol(control)

    if !haskey(q, :vec)
        prepareparameters!(q, keys(q))
    end

    U(q) = model(q, d)
    ∇U = q -> first(Zygote.gradient(U, q))
    datakeys = collect(keys(d))
    checkinitialization(q, U, ∇U)

    transformedparameterskeys = setdiff(keys(d), datakeys)

    if !haskey(d, :vec)
        prepareparameters!(d, transformedparameterskeys)
    end

    # store parameters + transfomred parameters
    ndim = q[:length]
    samplesdim = ndim + d[:length]
    I = control[:iterations] + control[:iterations_warmup]

    samples = zeros(I, samplesdim)
    assignparameters!(samples, 1, 0, q)
    assignparameters!(samples, 1, ndim, d)

    # TODO (ear) figure out their type structure
    # advance RandomNumbers RNG
    # and document this requirement.
    advance!(control[:rng], convert(UInt64, 1 << 50 * control[:chainid]))
    M = setmetric(control[:metric], ndim)
    psample = generatemomentum(control[:rng], ndim, M)

    ε = findepsilon(1.0, deepcopy(q), deepcopy(psample), U, ∇U, M, control)
    μ = control[:μ]
    εbar = 0.0
    sbar = 0.0
    xbar = 0.0

    W = WelfordState(zeros(ndim), zero(M), 0)
    openwindow, closewindow, lastwindow, windowstep = adaptionwindows(control)

    leaps = zeros(Int, I)
    treedepths = zeros(Int, I)
    energies = zeros(I)
    acceptstats = zeros(I)
    εs = zeros(I)
    divergences = falses(I)
    stepsizecounter = 0

    for i = 2:I
        updateq!(q, samples[i - 1, 1:ndim])
        p = generatemomentum(control[:rng], ndim, M)
        H0 = hamiltonian(q, p, U, M)
        H = H0

        qf, pf = deepcopy(q), deepcopy(p)
        qb, pb = deepcopy(q), deepcopy(p)
        qsample, psample = deepcopy(q), deepcopy(p)
        qpr, ppr = deepcopy(q), deepcopy(p)

        # Momentum and sharp momentum at forward end of forward subtree
        pff = similar(p)
        pff .= p
        psharpff = similar(p)
        psharpff .= control[:skewsymmetric] ? p : rhosharp(p, M) # dtau_dp

        # Momentum and sharp momentum at backward end of forward subtree
        pfb = similar(p)
        pfb .= p
        psharpfb = similar(p)
        psharpfb .= psharpff

        # Momentum and sharp momentum at forward end of backward subtree
        pbf = similar(p)
        pbf .= p
        psharpbf = similar(p)
        psharpbf .= psharpff

        # Momentum and sharp momentum at backward end of backward subtree
        pbb = similar(p)
        pbb .= p
        psharpbb = similar(p)
        psharpbb .= psharpff

        # Integrated momenta along trajectory
        rho = similar(p)
        rho .= p

        α = 0.0
        depth = 0
        nleapfrog = 0
        lsw = 0.0

        while depth < control[:maxtreedepth]

            rhof = zeros(length(rho))
            rhob = zeros(length(rho))
            lswsubtree = -Inf

            if rand(control[:rng]) > 0.5
                rhob .= rho
                pbf .= pff
                psharpbf .= psharpff

                qpr, ppr, validsubtree, nleapfrog, lswsubtree, α =
                    buildtree!(depth, qf, pf, psharpfb, psharpff, rhof, pfb, pff,
                               H0, 1 * ε, U, ∇U, M, nleapfrog, lswsubtree, α, control)
            else
                rhof .= rho
                pfb .= pbb
                psharpfb .= psharpbb

                qpr, ppr, validsubtree, nleapfrog, lswsubtree, α =
                    buildtree!(depth, qb, pb, psharpbf, psharpbb, rhob, pbf, pbb,
                              H0, -1 * ε, U, ∇U, M, nleapfrog, lswsubtree, α, control)
            end

            if !validsubtree
                divergences[i] = true
                break
            end
            depth += 1

            if lswsubtree > lsw
                qsample, psample = deepcopy(qpr), deepcopy(ppr)
            else
                if rand(control[:rng]) < exp(lswsubtree - lsw)
                    qsample, psample = deepcopy(qpr), deepcopy(ppr)
                end
            end

            lsw = logsumexp(lsw, lswsubtree)

            # Demand satisfication around merged subtrees
            rho .= rhob + rhof
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

        energies[i] = hamiltonian(qsample, psample, U, M)
        assignparameters!(samples, i, 0, qsample)
        assignparameters!(samples, i, ndim, d)
        treedepths[i] = depth
        leaps[i] = nleapfrog
        εs[i] = ε
        acceptstats[i] = α / nleapfrog

        if i < control[:iterations_warmup]
            ε, εbar, stepsizecounter, sbar, xbar =
                adaptstepsize(acceptstats[i], stepsizecounter, sbar, xbar, μ, control)

            if openwindow <= i <= lastwindow
                W = accmoments(W, samples[i, 1:ndim])
            end

            if i == closewindow
                # reset var
                M = samplevariance(W, control)
                W = WelfordState(zeros(ndim), zero(M), 0)

                # reset stepsize
                ε = findepsilon(ε, deepcopy(qsample), deepcopy(psample), U, ∇U, M, control)
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

    cleancontainer(q, [:vec :length])
    cleancontainer(d, [transformedparameterskeys... [:vec :length]])

    postwarmup = control[:iterations_warmup]+1:I
    return samples[postwarmup, :], Dict(:stepsize => εs,
                                        :leapfrog => leaps,
                                        :acceptstat => acceptstats,
                                        :treedepth => treedepths,
                                        :energy => energies,
                                        :divergent => divergences,
                                        :massmatrix => M)
end


# TODO separate q::Dict and p::Vector
function buildtree!(depth::Int, q::Dict, p::Vector{Float64},
                    psharpbeg::Vector{Float64}, psharpend::Vector{Float64},
                    rho::Vector{Float64}, pbeg::Vector{Float64}, pend::Vector{Float64},
                    H0::Float64, ε::Float64, U, ∇U, M::AbstractArray{Float64},
                    nleapfrog::Int, logsumweight::Float64, α::Float64, control)
    if depth == 0
        leapfrog!(q, p, ∇U, ε, M)
        qpr, ppr = deepcopy(q), deepcopy(p)
        nleapfrog += 1

        H = hamiltonian(qpr, ppr, U, M)
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

        psharpbeg .= control[:skewsymmetric] ? p : rhosharp(p, M) # dtau_dp
        psharpend .= psharpbeg

        rho .= rho + ppr
        pbeg .= ppr
        pend .= pbeg

        return qpr, ppr, !divergent, nleapfrog, logsumweight, α
    end

    lswinit = -Inf

    psharpinitend = similar(p)
    rhoinit = zeros(length(rho))
    pinitend = similar(p)

    qpr, ppr, validinit, nleapfrog, lswinit, α =
        buildtree!(depth - 1, q, p,
                  psharpbeg, psharpinitend, rhoinit, pbeg, pinitend,
                  H0, ε, U, ∇U, M, nleapfrog, lswinit, α, control)

    if !validinit
        return qpr, ppr, false, nleapfrog, logsumweight, α
    end

    lswfinal = -Inf

    psharpfinalbeg = similar(p)
    rhofinal = zeros(length(rho))
    pfinalbeg = similar(p)

    qfinalpr, pfinalpr, validfinal, nleapfrog, lswfinal, α =
        buildtree!(depth - 1, q, p,
                   psharpfinalbeg, psharpend, rhofinal, pfinalbeg, pend,
                   H0, ε, U, ∇U, M, nleapfrog, lswfinal, α, control)

    if !validfinal
        return qfinalpr, pfinalpr, false, nleapfrog, logsumweight, α
    end

    lswsubtree = logsumexp(lswinit, lswfinal)

    if lswfinal > lswsubtree
        qpr, ppr = deepcopy(qfinalpr), deepcopy(pfinalpr)
    else
        if rand(control[:rng]) < exp(lswfinal - lswsubtree)
            qpr, ppr = deepcopy(qfinalpr), deepcopy(pfinalpr)
        end
    end

    logsumweight = logsumexp(logsumweight, lswsubtree)

    rhosubtree = rhoinit + rhofinal
    rho .= rho + rhosubtree

    # Demand satisfaction around merged subtrees
    persist = stancriterion(psharpbeg, psharpend, rhosubtree)

    # Demand satisfaction between subtrees
    rhosubtree = rhoinit + pfinalbeg
    persist &= stancriterion(psharpbeg, psharpfinalbeg, rhosubtree)

    rhosubtree = rhofinal + pinitend
    persist &= stancriterion(psharpinitend, psharpend, rhosubtree)

    return qpr, ppr, persist, nleapfrog, logsumweight, α
end

function leapfrog!(q::Dict, p::Vector{Float64}, ∇U, ε::Float64, M::Array{Float64})
    updatep!(p, ∇U, q, 0.5 * ε)
    updateq!(q, ε * rhosharp(p, M); addself = true)
    updatep!(p, ∇U, q, 0.5 * ε)
    return
end

# TODO want this bad enough to mess with L' * ∇U(q)::Dict?
# function leapfrog!(q::Dict, p::Vector{Float64}, ∇U, ε::Float64,
#                    L::LowerTriangular{Float64,Matrix{Float64}})
#     p_ = p - 0.5 * ε * (L' * ∇U(q))
#     updateq!(q, ε * rhosharp(p_, L); addself = true)
#     p .= p_ - 0.5 * ε * (L' * ∇U(q))
#     return
# end

function hamiltonian(q::Dict, p::Vector{Float64}, U, M::Array{Float64})::Float64
    return U(q) + 0.5 * (p' * rhosharp(p, M))
end

function hamiltonian(q::Dict, p::Vector{Float64},
                     L::LowerTriangular{Float64,Matrix{Float64}})::Float64
    return U(q) + 0.5 * (p' * p)
end

function rhosharp(p::Vector{Float64}, M::Vector{Float64})::Vector{Float64}
    return M .* p
end

function rhosharp(p::Vector{Float64}, M::Matrix{Float64})::Vector{Float64}
    return M * p
end

function rhosharp(p::Vector{Float64},
                  L::LowerTriangular{Float64,Matrix{Float64}})::Vector{Float64}
    return L * p
end

function stancriterion(psharp_m::Vector{Float64}, psharp_p::Vector{Float64},
                       rho::Vector{Float64})::Bool
    return psharp_p' * rho > 0 && psharp_m' * rho > 0
end

function generatemomentum(rng, ndim::Int, M::Vector{Float64})::Vector{Float64}
    return randn(rng, ndim) ./ sqrt.(M)
end

function generatemomentum(rng, ndim::Int, M::Matrix{Float64})::Vector{Float64}
    return cholesky(Symmetric(M)).U \ randn(rng, ndim)
end

function generatemomentum(rng, ndim::Int,
                          L::LowerTriangular{Float64,Matrix{Float64}})::Vector{Float64}
    return randn(rng, ndim)
end


function findepsilon(ε::Float64, q::Dict, p::Vector{Float64}, U, ∇U,
                     M::AbstractArray{Float64}, control)::Float64
    ndim = q[:length]
    H0 = hamiltonian(q, p, U, M)

    leapfrog!(q, p, ∇U, ε, M)
    H = hamiltonian(q, p, U, M)
    if isnan(H)
        H = Inf
    end

    ΔH = H0 - H
    direction = ΔH > log(0.8) ? 1 : -1

    while true
        p = generatemomentum(control[:rng], ndim, M)
        H0 = hamiltonian(q, p, U, M)

        leapfrog!(q, p, ∇U, ε, M)
        H = hamiltonian(q, p, U, M)
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


function adaptionwindows(control)

    # TODO(ear) check, Madapt * termpercent > 20
    openwindow::Int = floor(control[:iterations_warmup] *
        control[:adaptinitpercent])

    lastwindow::Int = ceil(control[:iterations_warmup] *
        (1 - control[:adapttermpercent]))

    return openwindow, openwindow + control[:adaptwindowsize],
    lastwindow, control[:adaptwindowsize]
end


function initializesample(ndim::Int, U, ∇U, control)::Vector{Float64}
    q = zeros(ndim)
    initialized = false
    a = 0

    while a < control[:initattempts] && !initialized
        q = generateuniform(control[:rng], length(q), control[:initradius])

        lq = U(q)
        if isfinite(lq) && !isnan(lq)
            initialized = true
        end

        gq = sum(∇U(q))
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

abstract type AbstractWelfordState{S, T} end

struct WelfordState{S<:Vector{Float64}, T<:AbstractArray{Float64}} <: AbstractWelfordState{S, T}
    m::S
    s::T
    n::Int
end

function accmoments(ws::AbstractWelfordState, x::Vector{Float64})::WelfordState
    n = ws.n + 1
    d = similar(ws.m)
    M = similar(ws.m)
    @. d = x - ws.m
    @. M = ws.m + d / n
    S = updateCov(ws.s, d, x .- M)
    return WelfordState(M, S, n)
end

function updateCov(s::Vector{Float64}, d::Vector{Float64}, dp::Vector{Float64})
    return @. s + d * dp
end

function updateCov(s::AbstractArray{Float64}, d::Vector{Float64}, dp::Vector{Float64})
    return s .+ dp * d'
end

function samplevariance(ws::AbstractWelfordState{Vector{Float64}, Vector{Float64}},
                        control)
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

function samplevariance(ws::AbstractWelfordState{Vector{Float64}, Matrix{Float64}},
                        control)
    if ws.n > 1
        σ = similar(ws.s)
        @. σ = ws.s / (ws.n - 1)
        if control[:regularize]
            w = ws.n / (ws.n + 5.0)
            σ = w * σ + 1e-3 * (1 - w) * one(σ)
        end
        if control[:skewsymmetric]
            return cholesky(Symmetric(σ)).L
        end
        return σ
    end
    return one(ws.s)
end

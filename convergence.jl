# Most of this work is borrowed from
# https://github.com/stan-dev/rstan
# Copyright (C) 2012, 2013, 2014, 2015, 2016, 2017, 2018 Trustees of Columbia University
# Copyright (C) 2018, 2019 Aki Vehtari, Paul Bürkner

# References

# Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki
# Vehtari and Donald B. Rubin (2013). Bayesian Data Analysis, Third
# Edition. Chapman and Hall/CRC.

# Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, and
# Paul-Christian Bürkner (2019). Rank-normalization, folding, and
# localization: An improved R-hat for assessing convergence of
# MCMC. arXiv preprint arXiv:1903.08008

using Statistics
using FFTW
using Distributions

function autocovariance(x)
    N = length(x)
    Mt2 = 2 * fft_nextgoodsize(N)
    yc = x .- mean(x)
    yc = append!(yc, repeat([0], Mt2 - N))
    t = bfft(yc)
    ac = bfft(conj(t) .* t)
    return real(ac)[1:N] ./ (N * N * 2)
end

function autocorrelation(x)
    ac = autocovariance(x)
    return ac / ac[1]
end


function fft_nextgoodsize(N)
    N <= 2 && return 2

    while true
        m = N
        while mod(m, 2) == 0
            m /= 2
        end
        while mod(m, 3) == 0
            m /= 3
        end
        while mod(m, 5) == 0
            m /= 5
        end
        if m <= 1
            return N
        end
        N += 1
    end
end


function ess(x)

    niterations, nchains = size(x)

    (niterations < 3 || any(isnan.(x))) && return NaN
    any(isinf.(x)) && return NaN
    isconstant(x) && return NaN

    acov = mapslices(autocovariance, x, dims=1)
    chain_mean = mean(x, dims=1)
    mean_var = mean(acov[1, :]) * niterations / (niterations - 1)
    var_plus = mean_var * (niterations - 1) / niterations

    if nchains > 1
        var_plus += var(chain_mean)
    end

    rhohat = zeros(niterations)
    t = 0
    rhohat_even = 1.0
    rhohat[t + 1] = rhohat_even
    rhohat_odd = 1 - (mean_var - mean(acov[t + 2, :])) / var_plus
    rhohat[t + 2] = rhohat_odd

    while t < niterations - 5 &&
        !isnan(rhohat_even + rhohat_odd) &&
        rhohat_even + rhohat_odd > 0
        t += 2
        rhohat_even = 1 - (mean_var - mean(acov[t + 1, :])) / var_plus
        rhohat_odd = 1 - (mean_var - mean(acov[t + 2, :])) / var_plus

        if rhohat_even + rhohat_odd >= 0
            rhohat[t + 1] = rhohat_even
            rhohat[t + 2] = rhohat_odd
        end
    end

    max_t = t
    # this is used in the improved estimate
    if rhohat_even > 0
        rhohat[max_t + 1] = rhohat_even
    end

    # Geyer's initial monotone sequence
    t = 0
    while t <= max_t - 4
        t += 2
        if rhohat[t + 1] + rhohat[t + 2] > rhohat[t - 1] + rhohat[t]
            rhohat[t + 1] = (rhohat[t - 1] + rhohat[t]) / 2
            rhohat[t + 2] = rhohat[t + 1]
        end
    end

    ess = nchains * niterations
    # Geyer's truncated estimate
    # it's possible max_t == 0; 1:0 does not behave like in R
    τ = -1 + 2.0 * sum(rhohat[1:max(1, max_t)]) + rhohat[max_t + 1]
    # Improved estimate reduces variance in antithetic case
    τ = max(τ, 1.0 / log10(ess))
    return ess / τ
end

function rhat_impl(x)
    any(isnan.(x)) && return NaN
    any(isinf.(x)) && return NaN
    isconstant(x) && return NaN

    niterations, nchains = size(x)

    chain_mean = mean(x, dims=1)
    chain_var = var(x, dims=1)

    var_between = niterations * var(chain_mean)
    var_within = mean(chain_var)

    return sqrt((var_between / var_within + niterations - 1) / niterations)
end

function rhat(x)
    rhatbulk = rhat_impl()
end


function isconstant(x, tol=sqrt(eps(0.0)))
    return isapprox(minimum(x), maximum(x), rtol=tol)
end


function splitchains(x)
    niterations = size(x, 1)
    niterations < 2 && return x

    half = niterations / 2
    ub_lowerhalf::Int = floor(half)
    lb_secondhalf::Int = ceil(half + 1)
    return hcat(x[1:ub_lowerhalf, :], x[lb_secondhalf:end, :])
end

function fold(x)
    return abs.(x .- median(x))
end

function tiedrank(x)
    # Borrowed with slight modification from StatsBase
    # https://github.com/JuliaStats/StatsBase.jl/blob/master/src/ranking.jl

    # Copyright (c) 2012-2016: Dahua Lin, Simon Byrne, Andreas Noack,
    # Douglas Bates, John Myles White, Simon Kornblith, and other contributors.

    # Permission is hereby granted, free of charge, to any person obtaining
    # a copy of this software and associated documentation files (the
    # "Software"), to deal in the Software without restriction, including
    # without limitation the rights to use, copy, modify, merge, publish,
    # distribute, sublicense, and/or sell copies of the Software, and to
    # permit persons to whom the Software is furnished to do so, subject to
    # the following conditions:
    #
    # The above copyright notice and this permission notice shall be
    # included in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    # NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
    # LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
    # OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    # WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    if length(size(x)) > 1
        x = x[:]
    end

    n = length(x)
    p = sortperm(x)
    rks = zeros(n)

    if n > 0
        v = x[p[1]]

        s = 1  # starting index of current range
        e = 2  # pass-by-end index of current range
        while e <= n
            cx = x[p[e]]
            if cx != v
                # fill average rank to s : e-1
                ar = (s + e - 1) / 2
                for i = s : e-1
                    rks[p[i]] = ar
                end
                # switch to next range
                s = e
                v = cx
            end
            e += 1
        end

        # the last range (e == n+1)
        ar = (s + n) / 2
        for i = s : n
            rks[p[i]] = ar
        end
    end

    return rks
end

function zscale(x)
    r = tiedrank(x[:])
    N = Normal()
    z = quantile.(N, (r .- 0.5) ./ length(x[:]))
    if length(size(x)) > 1
        z = reshape(z, size(x))
    end
    return z
end

function rhat_basic(x, split = true)
    split && (x = splitchains(x))
    return rhat_impl(x)
end


function ess_basic(x, split = true)
    split && (x = splitchains(x))
    return ess(x)
end

function rhat(x)
    rhat_bulk = rhat_impl(zscale(splitchains(x)))
    rhat_tail = rhat_impl(zscale(splitchains(fold(x))))
    return max(rhat_bulk, rhat_tail)
end

function ess_bulk(x)
    return ess(zscale(splitchains(x)))
end

function ess_tail(x)
    I05 = x .<= quantile(x[:], 0.05)
    q05_ess = ess(splitchains(I05))
    I95 = x .<= quantile(x[:], 0.95)
    q95_ess = ess(splitchains(I95))
    return min(q05_ess, q95_ess)
end

function ess_quantile(x, prob::Real)
    @assert prob >= 0 && prob <= 1
    I = x .<= quantile(x[:], prob)
    return ess(splitchains(I))
end

function ess_mean(x)
    return ess(splitchains(x))
end

function ess_sq(x)
    return ess(splitchains(x .^ 2))
end

function ess_std(x)
    return min(ess_mean(x), ess_sq(x))
end

function mcse_quantile(x, prob::Real)
    ess = ess_quantile(x, prob)
    p = [0.1586553; 0.8413447]
    B = Beta(ess * prob + 1, ess * (1 - prob) + 1)
    a = quantile.(B, p)
    ssims = sort(x[:])
    S = length(ssims)
    th1 = ssims[convert(Int64, max(floor(a[1] * S), 1))]
    th2 = ssims[convert(Int64, min(ceil(a[2] * S), S))]
    return (th2 - th1) / 2
end

function mcse_mean(x)
    return std(x[:]) / sqrt(ess(x))
end

function mcse_std(x)
    ess_sd = ess_std(x)
    return std(x[:]) * sqrt(exp(1) * (1 - 1 / ess_sd) ^ (ess_sd - 1) - 1)
end

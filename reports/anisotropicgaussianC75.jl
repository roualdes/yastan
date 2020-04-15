using HDF5
using SparseArrays
using Statistics
include("../convergence.jl")
include("../hmc.jl")
include("../models.jl")

function main()
    R = 101
    D = vcat(1, range(2, 52, step=5))
    control = Dict(:iterations => 2000, :chains => 4)
    M = sparse([1.0 1.5; 1.5 4.0])

    for d in D
        S = Matrix(blockdiag([M for i in 1:d]...))
        f = mvgaussian_(zeros(2d), S)
        metric = Matrix(Diagonal(ones(2d)))
        essbulk = zeros(R)
        esstail = zeros(R)
        esssq = zeros(R)
        leapfrog = zeros(R)
        Qs = zeros(R, 3)
        println("Starting ", 2d, " dimensions")
        for r in 1:R
            samples, c = stan(f, 2d; M = metric, control = control)
            s = samples[:, :, 2]
            essbulk[r] = ess_bulk(s)
            esstail[r] = ess_tail(s)
            esssq[r] = ess_sq(s)
            leapfrog[r] = sum(c[:leapfrog])
            draws = reshape(s, :, 2)
            Qs[r, :] = quantile(draws[:, 2], [0.1 0.5 0.9])
        end
        println("Writing data")
        g = "anisogaussianC75/yastan/D$(2d)/"
        h5open("data.h5", "w") do file
            write(file, g * "essbulk", essbulk)
            write(file, g * "esstail", esstail)
            write(file, g * "esssq", esssq)
            write(file, g * "leapfrog", leapfrog)
            write(file, g * "q10", Qs[:, 1])
            write(file, g * "q50", Qs[:, 2])
            write(file, g * "q90", Qs[:, 3])
        end
    end
end

main()

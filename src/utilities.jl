function prepareq(q::Dict)
    q = convert(Dict{Any, Any}, q)
    q[:vec] = Dict()
    l = zeros(Int, length(q))
    idx = 1
    d = 0
    for (i, (k, v)) in enumerate(q)
        if k != :vec
            l[i] = length(v)
            jdx = idx + l[i] - 1
            q[:vec][k] = idx:jdx
            d += length(q[:vec][k])
            idx += l[i]
        end
    end
    q[:length] = d
    return q
end

function assignq!(v::Vector{Float64}, q::Dict)
    for k in keys(q[:vec])
        if k != :vec || k != :length
            idx = q[:vec][k]
            if typeof(q[k]) <: AbstractArray
                v[idx] .= vec(q[k])
            else
                v[idx] .= q[k]
            end
        end
    end
    return
end

function assignq!(v::Array{Float64}, q::Dict, row::Int = 1)
    for k in keys(q[:vec])
        if k != :vec || k != :length
            idx = q[:vec][k]
            if typeof(q[k]) <: AbstractArray
                v[row, idx] .= vec(q[k])
            else
                v[row, idx] .= q[k]
            end
        end
    end
    return
end

function updateq!(q::Dict, v::Vector{Float64}; addself = false)
    @assert length(v) == q[:length] "Can't update q with vector v, differing lengths."
    for k in keys(q[:vec])
        idx = q[:vec][k]
        if typeof(q[k]) <: AbstractArray
            tmpv = reshape(v[idx], size(q[k]))
            q[k] .= addself ? q[k] + tmpv : tmpv
        else
            q[k] = addself ? q[k] + v[idx[1]] : v[idx[1]]
        end
    end
    return
end

# TODO add tests
function updatep!(∇U, q, p, ε)
    @assert length(p) == q[:length] "Can't multiply ∇U(q) by p, differening lengths."
    gq = ∇U(q)
    for k in keys(q[:vec])
        idx = q[:vec][k]
        if typeof(q[k]) <: AbstractArray
            p[idx] .= p[idx] - ε * vec(gq[k])
        else
            p[idx] .= p[idx[1]] - ε * gq[k]
        end
    end
    return
end

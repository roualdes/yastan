function prepareq!(q::Dict)
    q[:vec] = Dict()
    l = zeros(Int, length(q))
    idx = 1
    d = 0
    for (i, (k, v)) in enumerate(q)
        if k == :vec
            continue
        end
        l[i] = length(v)
        jdx = idx + l[i] - 1
        q[:vec][k] = idx:jdx
        d += length(q[:vec][k])
        idx += l[i]
    end
    q[:length] = d
    return
end

function assignq!(v::Vector{Float64}, q::Dict)
    @assert length(v) == q[:length] "Can't assign q into vector v, differing lengths."
    for k in keys(q[:vec])
        idx = q[:vec][k]
        if typeof(q[k]) <: AbstractArray
            v[idx] .= vec(q[k])
        else
            v[idx[1]] = q[k]
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

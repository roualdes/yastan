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

# function prepareq(q::Dict)
#     # TODO would prefer not to make a copy of q
#     # necessitates creation of q as Dict{Symbol, Any}(...)
#     q = convert(Dict{Any, Any}, q)
#     q[:vec] = Dict()
#     l = zeros(Int, length(q))
#     idx = 1
#     d = 0
#     for (i, (k, v)) in enumerate(q)
#         if k != :vec
#             l[i] = length(v)
#             jdx = idx + l[i] - 1
#             q[:vec][k] = idx:jdx
#             d += length(q[:vec][k])
#             idx += l[i]
#         end
#     end
#     q[:length] = d
#     return q
# end

function prepareparameters!(D::Dict, ks)
    @assert typeof(D) <: Dict{Symbol, Any} "$D has wrong type.  Try creating $D as Dict{Symbol, Any}(...)."
    @assert !(:vec in ks) && !(:length in ks) "Can't index keys :vec or :length."
    D[:vec] = Dict{Symbol, Any}()
    l = zeros(Int, length(ks))
    idx = 1
    ndims = 0
    for (i, k) in enumerate(ks)
        if k != :vec
            l[i] = length(D[k])
            jdx = idx + l[i] - 1
            D[:vec][k] = idx:jdx
            ndims += length(D[:vec][k])
            idx += l[i]
        end
    end
    D[:length] = ndims
    return
end

function parameternames(D::Dict)
    a = Array{String}(undef, D[:length])
    for k in keys(D[:vec])
        for i in D[:vec][k]
            a[i] = string(k) * "_" * string(i)
        end
    end
    return a
end

function assignq!(v::Vector{Float64}, D::Dict)
    for k in keys(D[:vec])
        idx = D[:vec][k]
        if typeof(D[k]) <: AbstractArray
            v[idx] .= vec(D[k])
        else
            v[idx] .= D[k]
        end
    end
    return
end

# function assignq!(a::Array{Float64}, row::Int, q::Dict)
#     for k in keys(q[:vec])
#         if k != :vec || k != :length
#             idx = q[:vec][k]
#             if typeof(q[k]) <: AbstractArray
#                 a[row, idx] .= vec(q[k])
#             else
#                 a[row, idx] .= q[k]
#             end
#         end
#     end
#     return
# end

function assignparameters!(a::Array{Float64}, row::Int, offset::Int, D::Dict)
    for k in keys(D[:vec])
        idx = D[:vec][k]
        if typeof(D[k]) <: AbstractArray
            a[row, idx .+ offset] .= vec(D[k])
        else
            a[row, idx .+ offset] .= D[k]
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
# TODO should be updatep!(p, ∇U, q, ε)
function updatep!(p::Vector{Float64}, ∇U, q::Dict, ε::Float64)
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


function setmetric(metric::String, ndim::Int)::AbstractArray{Float64}
    @assert metric in ["diag", "dense", "skewsymmetric"] "Metric $metric not understood.  Options are diag, dense, or skewsymmetric."

    if metric == "skewsymmetric"
        M = cholesky(Symmetric(Matrix(Diagonal(ones(ndim))))).L
    elseif metric == "dense"
        M = Matrix(Diagonal(ones(ndim)))
    else
        M = ones(ndim)
    end

    return M
end


function checkinitialization(q, U, ∇U)
    lq = U(q)
    @assert isfinite(lq) && !isnan(lq) "Poor initialization, model(q, d) is not finite or is nan."
    p = zeros(q[:length])
    updatep!(p, ∇U, q, 1.0)
    gq = sum(abs.(p))
    @assert isfinite(gq) && !isnan(gq) "Poor initialization, gradient(model(q, d)) is not finite or is nan."
    return
end


function cleancontainer(D::Dict, ks)
    for k in ks
        delete!(D, k)
    end
    return
end

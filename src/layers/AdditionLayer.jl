include("LayerBase.jl")
type AdditionLayer <: Layer
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init :: Bool

    xs :: Array{Array{Float64}}
    y :: Array{Float64}
    dldy :: Array{Float64}
    dldxs:: Array{Array{Float64}}

    function AdditionLayer()
        return new(Layer[], Layer[], false, Float64[][], Float64[], Float64, Float64[][])
    end
end

function init(l::AdditionLayer, ps::Union{Array{Layer}}, config::Dict{String, Any}; kwargs...)
    out_size = getOutputSize(ps[1])
    for p in ps
        @assert getOutputSize(p) == out_size
        push!(p.children, l)
        push!(l.parents, p)
    end
    parents_size = length(ps)

    l.xs = Array{Array{Float64}(out_size), parents_size}
    l.y = Array{Float64}(out_size)
    l.dldxs = Array{Array{Float64}(out_size), parents_size}
    l.dldy = Array{Float64}(out_size)

    l.has_init = true
end

function update(l::AdditionLayer, input_size::Tuple;)
    l.xs = Array{Array{Float64}(input_size), parents_size}
    l.y = Array{Float64}(input_size)
    l.dldxs = Array{Array{Float64}(input_size), parents_size}
    l.dldy = Array{Float64}(input_size)
end

function forward(l::AdditionLayer, xs::Union{Array{SubArray{Float64}}, Array{Array{Float64}}}; kwargs...)
    if size(xs) != size(l.xs)
        update(l, size(xs[1]))
    end
    l.xs = xs
    l.y = zeros(l.y)
    for x in xs
        broadcast!(+, l.y, l.y, x)
    end
    return l.y
end

function backward(l::AdditionLayer, DLDY::Union{Array{Float64}, SubArray{Float64}};)
    @assert size(l.dldy) == size(DLDY)
    ndldxs = Array{Array{float64}}
    l.dldy = DLDY
    for n in length(l.dldxs)
        push!(ndldxs, DLDY)
    end
    l.dldxs = ndldxs
    return l.dldxs
end

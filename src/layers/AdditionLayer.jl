include("LayerBase.jl")
type AdditionLayer <: Layer
    base    :: LayerBase

    function AdditionLayer(prevs::Array{<:Layer}; kwargs...)
        layer =  new(LayerBase())
        init(layer, prevs; kwargs...)
        layer
    end
end

function init(l::AdditionLayer, ps::Union{Array{<:Layer}}; kwargs...)
    out_size = getOutputSize(ps[1])
    for p in ps
        @assert getOutputSize(p) == out_size
        push!(p.base.children, l)
        push!(l.base.parents, p)
    end

    parents_size = length(ps)

    l.base.y = Array{Float64}(out_size)
    l.base.dldx = Array{Float64}(out_size)

    l.base.has_init = true
end

function update(l::AdditionLayer, input_size::Tuple;)
    parents_size = length(l.base.parents)

    l.base.y = Array{Float64}(input_size)
    l.base.dldx = Array{Float64}(input_size)
end

function forward(l::AdditionLayer;kwargs...)
    xs = [l.base.y for l in l.base.parents]
    l.base.y = zeros(l.base.y)
    for i=1:size(xs)[1]
        broadcast!(+, l.base.y, l.base.y, xs[i])
    end
end

function backward(l::AdditionLayer, DLDY::Union{Array{Float64}, SubArray{Float64}};)
    @assert size(l.base.dldx) == size(DLDY)
    l.base.dldx = DLDY
    return l.dldx
end

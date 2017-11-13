include("LayerBase.jl")
type AdditionLayer <: Layer
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init :: Bool

    y     :: Array{Float64}
    dldx  :: Array{Float64}

    function AdditionLayer()
        return new(Layer[], Layer[], false, Float64[], Float64[], Float64[], Float64[])
    end

    function AdditionLayer(prevs::Array{<:Layer}, config::Union{Dict{String,Any},Void}=nothing)
        layer =  new(Layer[], Layer[], false, Float64[], Float64[])
        init(layer, prevs, config)
        layer
    end
end

function init(l::AdditionLayer, ps::Union{Array{<:Layer}}, config::Union{Dict{String,Any},Void}; kwargs...)
    out_size = getOutputSize(ps[1])
    for p in ps
        @assert getOutputSize(p) == out_size
        push!(p.children, l)
        push!(l.parents, p)
    end

    parents_size = length(ps)

    l.y = Array{Float64}(out_size)
    l.dldx = Array{Float64}(out_size)

    l.has_init = true
end

function update(l::AdditionLayer, input_size::Tuple;)
    parents_size = length(l.parents)

    l.y = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
end

function forward(l::AdditionLayer;kwargs...)
    xs = [l.y for l in l.parents]
    l.y = zeros(l.y)
    for i=1:size(xs)[1]
        broadcast!(+, l.y, l.y, xs[i])
    end
    return l.y
end

function backward(l::AdditionLayer, DLDY::Union{Array{Float64}, SubArray{Float64}};)
    @assert size(l.dldx) == size(DLDY)
    l.dldx = DLDY
    return l.dldx
end

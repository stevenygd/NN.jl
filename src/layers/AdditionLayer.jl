include("LayerBase.jl")
type AdditionLayer <: Layer
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init :: Bool

    xs    :: Array{Float64}
    y     :: Array{Float64}
    dldy  :: Array{Float64}
    dldxs :: Array{Float64}

    function AdditionLayer()
        return new(Layer[], Layer[], false, Float64[], Float64[], Float64[], Float64[])
    end

    function AdditionLayer(prevs::Array{<:Layer}, config::Dict{String, Any})
        layer =  new(Layer[], Layer[], false, Float64[], Float64[], Float64[], Float64[])
        init(layer, prevs, config)
        layer
    end
end

function init(l::AdditionLayer, ps::Union{Array{<:Layer}}, config::Dict{String, Any}; kwargs...)
    out_size = getOutputSize(ps[1])
    for p in ps
        @assert getOutputSize(p) == out_size
        push!(p.children, l)
        push!(l.parents, p)
    end

    parents_size = length(ps)
    size_array = [i for i in out_size] 
    push!(size_array, parents_size)
    size_tuple = Tuple(size_array)

    l.xs = Array{Float64}(size_tuple)
    l.y = Array{Float64}(out_size)
    l.dldxs = Array{Float64}(size_tuple)
    l.dldy = Array{Float64}(out_size)

    l.has_init = true
end

function update(l::AdditionLayer, input_size::Tuple;)
    parents_size = length(l.parents)
    size_array = [i for i in input_size] 
    push!(size_array, parents_size)
    size_tuple = Tuple(size_array)

    l.xs = Array{Float64}(size_tuple)
    l.y = Array{Float64}(input_size)
    l.dldxs = Array{Float64}(size_tuple)
    l.dldy = Array{Float64}(input_size)
end

function forward(l::AdditionLayer, xs::Union{SubArray{Float64}, Array{Float64}}; kwargs...)
    if size(xs) != size(l.xs)
        update(l, size(xs[1]))
    end
    l.xs = xs
    l.y = zeros(l.y)
    println(l.y)
    for i in 1:size(l.xs)[3]
        x = xs[:, :, i]
        broadcast!(+, l.y, l.y, x)
        println(l.y)
    end
    return l.y
end

function backward(l::AdditionLayer, DLDY::Union{Array{Float64}, SubArray{Float64}};)
    @assert size(l.dldy) == size(DLDY)
    ndldxs = zeros(l.dldxs)
    l.dldy = DLDY
    for n in 1:size(l.dldxs)[3]
        ndldxs[:, :, n] = l.dldy
    end
    l.dldxs = ndldxs
    return l.dldxs
end

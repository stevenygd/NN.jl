include("LayerBase.jl")
type AdditionLayer <: Layer
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init :: Bool

    xs    :: Array{Array{Float64}, 1}
    y     :: Array{Float64}
    dldxs :: Array{Array{Float64}, 1}
    dldy  :: Array{Float64}

    function AdditionLayer()
        return new(Layer[], Layer[], false, Float64[], Float64[], Float64[], Float64[])
    end

    function AdditionLayer(prevs::Array{<:Layer}, config::Dict{String, Any})
        layer =  new(Layer[], Layer[], false, Array{Float64}[], Float64[], Array{Float64}[], Float64[])
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

    l.xs = Array{Array{Float64, length(out_size)}, 1}()
    l.y = Array{Float64}(out_size)
    l.dldxs = Array{Array{Float64, length(out_size)}, 1}()
    l.dldy = Array{Float64}(out_size)

    l.has_init = true
end

function update(l::AdditionLayer, input_size::Tuple;)
    parents_size = length(l.parents)

    l.xs = Array{Array{Float64, length(input_size)}, 1}()
    l.y = Array{Float64}(input_size)
    l.dldcs = Array{Array{Float64, length(input_size)}, 1}()
    l.dldy = Array{Float64}(input_size)
end

function forward(l::AdditionLayer;kwargs...)
    xs = [l.x for l in l.parents]
    l.y = zeros(l.y)
    for i=1:size(xs)[1]
        broadcast!(+, l.y, l.y, xs[i])
    end
    return l.y
end

# function forward(l::AdditionLayer, xs::Union{SubArray{Float64}, Array{Float64}}; kwargs...)
#     if size(xs) != size(l.xs)
#         update(l, size(xs[1]))
#     end
#     l.xs = xs
#     l.y = zeros(l.y)
#     println(l.y)
#     for i in 1:size(l.xs)[3]
#         x = xs[:, :, i]
#         broadcast!(+, l.y, l.y, x)
#         println(l.y)
#     end
#     return l.y
# end

function backward(l::AdditionLayer, DLDY::Union{Array{Float64}, SubArray{Float64}};)
    @assert size(l.dldy) == size(DLDY)
    ndldxs = Array{Array{Float64, length(size(l.dldy))}, 1}()
    l.dldy = DLDY
    for n in 1:size(l.parents)[1]
        push!(ndldxs, dldy)
    end
    l.dldxs = ndldxs
    return l.dldxs
end

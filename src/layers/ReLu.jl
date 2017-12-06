# Define the ReLu layers
type ReLu <: Nonlinearity
    base     :: LayerBase

    alpha    :: Float64
    x        :: Array{Float64}
    dldy     :: Array{Float64}

    function ReLu(prev::Layer; alpha::Float64 = 1.0, kwargs...)
        @assert alpha ≥ 0.
        layer = new(LayerBase(), alpha, Float64[], Float64[])
        connect(layer, [prev])
        init(layer, getOutputSize(prev); kwargs...)
        layer
    end

    function ReLu(config::Dict{String,Any}; alpha::Float64 = 1.0, kwargs...)
        @assert alpha ≥ 0.
        layer = new(LayerBase(), alpha, Float64[], Float64[])
        @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
        init(layer, out_size; kwargs...)
        layer
    end
end

function init(l::ReLu, out_size::Tuple; kwargs...)
    @assert length(l.base.parents) == 1
    l.x = Array{Float64}(out_size)
    l.base.y = Array{Float64}(out_size)
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(out_size)
    l.dldy = Array{Float64}(out_size)
end

function update(l::ReLu, input_size::Tuple;)
    l.x = Array{Float64}(input_size)
    l.base.y = Array{Float64}(input_size)
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    l.dldy = Array{Float64}(input_size)
end

function forward(l::ReLu, X::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    if size(l.x) != size(X)
        update(l, size(X))
    end
    l.x = X
    broadcast!(max, l.base.y, X, 0.)
    broadcast!(*,   l.base.y, l.base.y, l.alpha)
    return l.base.y
end

function backward(l::ReLu, DLDY::Array{Float64}; kwargs...)
    if size(l.x) != size(DLDY)
        println(size(DLDY))
        println(size(l.x))
    end
    @assert size(l.x) == size(DLDY)
    # if size(l.base.dldx, 1) != size(DLDY, 1)
    #     l.base.dldx = Array{Float64}(size(DLDY))
    # end
    l.dldy = DLDY
    parent_id = l.base.parents[1].base.id
    broadcast!(>, l.base.dldx[parent_id], l.x, 0.)        # l.base.dldx = l.x .> 0.
    broadcast!(*, l.base.dldx[parent_id], l.base.dldx[parent_id], l.alpha)    # l.base.dldx = l.base.dldx * alpha
    broadcast!(*, l.base.dldx[parent_id], l.base.dldx[parent_id], DLDY)
end

# l = ReLu()
# X = rand(1000, 500)
# Y = rand(1000, 500)
# println("Compile the method for the first time...")
# @time init(l, nothing, Dict{String, Any}("batch_size" => 1000, "input_size" => [500]))
# @time forward(l,X)
# @time backward(l,Y)
#
# println("Start profiling...")
# print("Forward:")
# @time begin
#   for i = 1:10
#     forward(l,X)
#   end
# end
#
# @time begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end
#
# print("Backward")
# @time begin
#   for i = 1:10
#     backward(l,Y)
#   end
# end
#
# @time begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end

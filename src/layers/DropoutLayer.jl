# Abtract out DropOut
# include("LayerBase.jl")
type DropoutLayer <: RegularizationLayer
    has_init    :: Bool
    p           :: Float64
    last_drop   :: Array{Float64}
    x           :: Array{Float64}
    y           :: Array{Float64}
    dldx        :: Array{Float64}
    dldy        :: Array{Float64}

    function DropoutLayer(p::Float64)
        @assert abs(p - 1.) >  1e-4 # Basically [p] couldn't be 1
        return new(false, p, Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

function init(l::DropoutLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    # TODO: currently I only accept Single dimensional dropout
    if p == nothing
        # [l] is the first layer, batch_size used default network batch_size
        # and input_size should be single dimensional (i.e. vector)
        @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
    else
        out_size = getOutputSize(p)
        @assert length(out_size) == 2 # TODO: maybe a friendly error message?
    end
    N, D = out_size
    l.last_drop = Array{Float64}(N, D)
    l.x         = Array{Float64}(N, D)
    l.y         = Array{Float64}(N, D)
    l.dldy      = Array{Float64}(N, D)
    l.dldx      = Array{Float64}(N, D)
    l.has_init  = true
end

function update(l::DropoutLayer, input_size::Tuple;)
    # Reinitialize the memory due to the updated of the batch_size
    # Couldn't change the input and output size, only the bath size
    # the outter dimension must be the same, so that we don't need
    # to reinitialize the weights and bias
    @assert length(input_size) == 2
    l.last_drop = Array{Float64}(input_size)
    l.x         = Array{Float64}(input_size)
    l.y         = Array{Float64}(input_size)
    l.dldy      = Array{Float64}(input_size)
    l.dldx      = Array{Float64}(input_size)

    # println("DropoutLayer update:\n\tInput:$(size(l.x))\n\tOutput:$(size(l.y))")
end

function forward(l::DropoutLayer, x::Union{SubArray{Float64,2},Array{Float64,2}}; deterministics=false)
    # Need to rescale the inputs so that the expected output mean will be the same
    if size(x, 1) != size(l.x, 1)
        update(l, size(x))
    end
    l.x  = x
    rand!(l.last_drop)
    if ! deterministics
        broadcast!(>, l.last_drop, l.last_drop, l.p)
        broadcast!(/, l.last_drop, l.last_drop, 1-l.p)
    end
    broadcast!(*, l.y, l.last_drop, l.x)
    return l.y
end

# Donot annotate DLDY since it could be subarray
function backward(l::DropoutLayer, DLDY::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    @assert size(DLDY)[2] == size(l.last_drop)[2] &&
            size(DLDY)[1] == size(l.x)[1]
    l.dldy = DLDY
    broadcast!(*, l.dldx, l.dldy, l.last_drop)
    return l.dldx
end

function getInputSize(l::DropoutLayer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initizalized. But input shapes wanted.")
    end
    return size(l.x)
end

function getOutputSize(l::DropoutLayer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initizalized. But output shapes wanted.")
    end
    return size(l.y)
end

# l = DropoutLayer(0.3)
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

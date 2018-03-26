# Define the Fully Connected layers
# include("LayerBase.jl")
type FullyConnected <: LearnableLayer
    base    :: LayerBase
    x       :: Array{Float64}
    dldy    :: Array{Float64}
    dldx_cache :: Array{Float64}

    init_type :: String
    fan_in        :: Int
    fan_out :: Int
    W         :: Array{Float64}
    velc      :: Array{Float64}
    grad      :: Array{Float64}

    function FullyConnected(prev::Layer, fan_out::Int; init_type="He")
        i, o = 1, fan_out
        layer = new(LayerBase(), Float64[], Float64[], Float64[], init_type,
            i, o, randn(i+1,o), zeros(i+1, o), zeros(i+1, o))
        out_size = getOutputSize(prev)
        connect(layer, [prev])
        @assert length(layer.base.parents) == 1
        init(layer, out_size)
        layer
    end

    function FullyConnected(config, fan_out::Int; init_type="He")
        i, o = 1, fan_out
        layer = new(LayerBase(), Float64[], Float64[], Float64[], init_type,
            i, o, randn(i+1,o), zeros(i+1, o), zeros(i+1, o))
        @assert length(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
        init(layer, out_size)
        layer
    end

    # Create a copy of another FC layer; used for multi threading
    # function FullyConnected(main::FullyConnected)
    #     layer = new(LayerBase(), Float64[], Float64[], Float64[], "",
    #         0, 0, randn(1,1), zeros(1, 1), zeros(1, 1))
    #     layer.base.y = copy(main.base.y)
    #     layer.base.dldx = copy(main.base.dldx)
    #     layer.x = copy(main.x)
    #     layer.dldy = copy(main.dldy)
    #     layer.init_type = main.init_type
    #     layer.i = copy(main.i)
    #     layer.fan_out = copy(main.fan_out)
    #     layer.W = main.W
    #     layer.velc = copy(main.velc)
    #     layer.grad = copy(main.grad)
    #     layer
    # end
end

verbose = 0

function init(l::FullyConnected, out_size::Tuple; kwargs...)
    """
    [l]         the layer to be initialized
    [out_size]  the size of the output matrix
    """


    batch_size, l.fan_in = out_size

    # Get enough information, now preallocate the memory
    l.x     = Array{Float64}(batch_size, l.fan_in + 1)
    l.base.y = Array{Float64}(batch_size, l.fan_out)
    l.dldy  = Array{Float64}(batch_size, l.fan_out)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(batch_size, l.fan_in + 1)
    end
    l.velc  = zeros(l.fan_in + 1,    l.fan_out)
    l.grad  = zeros(l.fan_in + 1,    l.fan_out)

    # Pull out the output size
    if l.init_type == "Uniform"
        local a    = sqrt(12. / (l.fan_in +l.fan_out))
        l.W = rand(l.fan_in+1,l.fan_out)* 2 * a - a
    elseif l.init_type == "He"
        σ = sqrt(2. / l.fan_in)
        l.W  = randn(l.fan_in+1, l.fan_out) * σ
    elseif l.init_type == "Glorot"
        σ = sqrt(2. / (l.fan_in + l.fan_out))
        l.W  = randn(l.fan_in+1, l.fan_out) * σ
    elseif l.init_type == "Random"
        l.W = rand(l.fan_in+1, l.fan_out) - 0.5
    end
    # l.W[l.fan_in+1,:] = zeros(fan_out)

end

function update(l::FullyConnected, input_size::Tuple;)
    # Reinitialize the memory due to the updated of the batch_size
    # Couldn't change the input and output size, only the bath size
    # the outter dimension must be the same, so that we don't need
    # to reinitialize the weights and bias
    @assert length(input_size) == 2 && size(l.x, 2) == size(l.x, 2)
    batch_size = input_size[1]
    l.x      = Array{Float64}(batch_size, l.fan_in + 1)
    l.base.y = Array{Float64}(batch_size, l.fan_out)
    l.dldy   = Array{Float64}(batch_size, l.fan_out)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id]  = Array{Float64}(batch_size, l.fan_in + 1)
    end
    # println("FullyConnected update:\n\tInput:$(size(l.x))\n\tOutput:$(size(l.y))")
end

function forward(l::FullyConnected, X::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    # X      : NxI matrix, N is the mini batch size, I is the input size
    # Output : NxO matrix
    @assert size(X)[2] == l.fan_in

    # update the batch_size, need to re-allocate the memory
    if size(X, 1) != size(l.x, 1)
        update(l, size(X))
    end

    # Pad one at the end of the vector
    l.x[:,1:l.fan_in] = X
    l.x[:,l.fan_in+1] = 1

    # Multiplication inplaces
    A_mul_B!(l.base.y, l.x, l.W)
    return l.base.y
end

function backward(l::FullyConnected, DLDY::Array; kwargs...)
    @assert size(DLDY,2) == size(l.W,2)
    l.dldy = DLDY
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = A_mul_Bt(l.dldy, l.W)[:, 1:end-1]
    end
    At_mul_B!(l.grad, l.x, l.dldy)
end

function getGradient(l::FullyConnected)
  return Array[l.grad]
end

function getParam(l::FullyConnected)
    return Array[l.W]
end

function setParam!(l::FullyConnected, theta)
    @assert size(l.W) == size(theta[1])
    # broadcast!(-, l.velc, theta, l.W)
    # l.velc = theta[1] - l.W
    l.W = theta[1]
    # copy!(l.W, theta[1])
end

function getVelocity(l::FullyConnected)
    return Array[l.velc]
end

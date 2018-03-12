# Define the Fully Connected layers
# include("LayerBase.jl")
using FixedPointDecimals

type FullyConnectedFixed <: LearnableLayer
    base    :: LayerBase
    x       :: Array{Float64}
    dldy    :: Array{Float64}
    dldx_cache :: Array{Float64}

    init_type :: String
    fan_in    :: Int
    fan_out   :: Int
    W         :: Array{FixedDecimal{Int, 4}}
    velc      :: Array{Float64}
    grad      :: Array{Float64}

    function FullyConnectedFixed(prev::Layer, fan_out::Int; init_type="He")
        i, o = 1, fan_out
        layer = new(LayerBase(), Float64[], Float64[], Float64[], init_type,
            i, o, FixedDecimal{Int, 4}[], zeros(i+1, o), zeros(i+1, o))
        out_size = getOutputSize(prev)
        connect(layer, [prev])
        @assert length(layer.base.parents) == 1
        init(layer, out_size)
        layer
    end

    function FullyConnectedFixed(config, fan_out::Int; init_type="He")
        i, o = 1, fan_out
        layer = new(LayerBase(), Float64[], Float64[], Float64[], init_type,
            i, o, FixedDecimal{Int, 4}[], zeros(i+1, o), zeros(i+1, o))
        @assert length(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
        init(layer, out_size)
        layer
    end
end

function init(l::FullyConnectedFixed, out_size::Tuple; kwargs...)
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
    l.W = Array{FixedDecimal{Int, 4}}(l.fan_in+1, l.fan_out)
    if l.init_type == "Uniform"
        local a    = sqrt(12. / (l.fan_in +l.fan_out))
        W = rand(l.fan_in+1,l.fan_out) * 2 * a - a
    elseif l.init_type == "He"
        σ = sqrt(2. / l.fan_in)
        W  = randn(l.fan_in+1, l.fan_out) * σ
    elseif l.init_type == "Glorot"
        σ = sqrt(2. / (l.fan_in + l.fan_out))
        W  = randn(l.fan_in+1, l.fan_out) * σ
    elseif l.init_type == "Random"
        W = rand(l.fan_in+1, l.fan_out) - 0.5
    end
    for i=1:length(W)
        l.W[i] = FixedDecimal{Int, 4}(W[i])
    end
end

function update(l::FullyConnectedFixed, input_size::Tuple;)
    @assert length(input_size) == 2 && size(l.x, 2) == size(l.x, 2)
    batch_size = input_size[1]
    l.x      = Array{Float64}(batch_size, l.fan_in + 1)
    l.base.y = Array{Float64}(batch_size, l.fan_out)
    l.dldy   = Array{Float64}(batch_size, l.fan_out)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id]  = Array{Float64}(batch_size, l.fan_in + 1)
    end
end

function forward(l::FullyConnectedFixed, X::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    @assert size(X)[2] == l.fan_in

    if size(X, 1) != size(l.x, 1)
        update(l, size(X))
    end

    # Pad one at the end of the vector
    l.x[:,1:l.fan_in] = X
    l.x[:,l.fan_in+1] = 1

    # Multiplication inplaces
    l.base.y = l.x * l.W
    # A_mul_B!(l.base.y, l.x, l.W)
    return l.base.y
end

function backward(l::FullyConnectedFixed, DLDY::Array; kwargs...)
    @assert size(DLDY,2) == size(l.W,2)
    l.dldy = DLDY
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = A_mul_Bt(l.dldy, l.W)[:, 1:end-1]
    end
    # At_mul_B!(l.grad, l.x, l.dldy)
    l.grad = l.x' * l.dldy
end

function getGradient(l::FullyConnectedFixed)
  return Array[l.grad]
end

function getParam(l::FullyConnectedFixed)
    return Array[l.W]
end

function setParam!(l::FullyConnectedFixed, theta)
    @assert size(l.W) == size(theta[1])
    # broadcast!(-, l.velc, theta, l.W)
    # l.velc = theta[1] - l.W
    for i=1:length(theta[1])
        l.W[i] = FixedDecimal{Int, 4}(theta[1][i])
    end
    # copy!(l.W, theta[1])
end

function getVelocity(l::FullyConnectedFixed)
    return Array[l.velc]
end

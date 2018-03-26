# Define the Fully Connected layers
include("LayerBase.jl")
include("../fixed_point_number/BlockFixedArray.jl")
type BlockFixedFC{T} <: LearnableLayer where {T<:Signed}
    base    :: LayerBase
    x       :: Union{Array{Float64}, BlockFixedArray{T}}
    dldy    :: BlockFixedArray{T}

    init_type :: String
    fan_in    :: Int
    fan_out   :: Int
    W         :: BlockFixedArray{T}
    velc      :: BlockFixedArray{T}
    grad      :: BlockFixedArray{T}

    function FullyConnected(prev::Layer, σ::Real, fan_out::Int; init_type="He")
        i, o = 1, fan_out
        σ = 2.0^(-12)
        layer = new(LayerBase(), Float64[], BlockFixedArray{T}(σ), init_type,
                    i, o, BlockFixedArray{T}(σ), BlockFixedArray{T}(σ), BlockFixedArray{T}(σ))
        out_size = getOutputSize(prev)
        connect(layer, [prev])
        @assert length(layer.base.parents) == 1
        init(layer, out_size)
        layer
    end

end

function init(σ::Real, l::BlockFixedFC{T}, out_size::Tuple; kwargs...) where {T<:Signed}
    """
    [l]         the layer to be initialized
    [out_size]  the size of the output matrix
    """


    batch_size, l.fan_in = out_size

    # Get enough information, now preallocate the memory
    l.x      = BlockFixedArray{T}(σ, batch_size, l.fan_in + 1)
    l.base.y = BlockFixedArray{T}(σ, batch_size, l.fan_out)
    l.dldy   = BlockFixedArray{T}(σ, batch_size, l.fan_out)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(batch_size, l.fan_in + 1)
    end
    l.velc  = BlockFixedArray{T}(σ, l.fan_in + 1, l.fan_out)
    l.grad  = BlockFixedArray{T}(σ, l.fan_in + 1, l.fan_out)

    # Pull out the output size
    if l.init_type == "Uniform"
        local a    = sqrt(12. / (l.fan_in +l.fan_out))
        l.W = rand(T, σ, l.fan_in+1,l.fan_out)* 2 * a - a
    elseif l.init_type == "He"
        σ = sqrt(2. / l.fan_in)
        l.W  = randn(T, σ, l.fan_in+1, l.fan_out) * σ
    elseif l.init_type == "Glorot"
        σ = sqrt(2. / (l.fan_in + l.fan_out))
        l.W  = randn(T, σ, l.fan_in+1, l.fan_out) * σ
    elseif l.init_type == "Random"
        l.W = rand(T, σ, l.fan_in+1, l.fan_out) - 0.5
    end
    l.W.arr[l.fan_in+1,:] = zeros(T, fan_out)
end

function getGradient(l::FullyConnected)
  return Array[l.grad]
end

function getParam(l::FullyConnected)
    return Array[l.W]
end

function setParam!(l::FullyConnected, theta)
    @assert size(l.W) == size(theta[1])
    l.W = theta[1]
end

function getVelocity(l::FullyConnected)
    return Array[l.velc]
end

# Define the Fully Connected layers
include("LayerBase.jl")
include("../fixed_point_number/BlockFixedArray.jl")
type BlockFixedFC{T} <: LearnableLayer where {T<:Signed}
    base    :: LayerBase
    x       :: Union{Array{Float64}, BlockFixedArray{T}}
    dldy    :: BlockFixedArray{T}
    init_type :: String

    x_low :: Bool
    nonlinearity :: String
    σ :: Real
    fan_in    :: Int
    fan_out   :: Int

    W         :: BlockFixedArray{T}
    velc      :: BlockFixedArray{T}
    grad      :: BlockFixedArray{T}

    function BlockFixedFC(prev::Layer, fan_out::Int; σ = 2.0^(-12), init_type="He", x_low = true, nonlinearity="ReLu")
        i, o = 1, fan_out
        layer = new(LayerBase(), Float64[], BlockFixedArray{T}(σ), init_type,
                    x_low, nonlinearity, σ, i, o,
                    BlockFixedArray{T}(σ), BlockFixedArray{T}(σ), BlockFixedArray{T}(σ)
                    )
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

function forward(l::BlockFixedFC{T}, X::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...) where {T<:Signed}
    # X      : NxI matrix, N is the mini batch size, I is the input size
    # Output : NxO matrix
    @assert size(X)[2] == l.fan_in

    # update the batch_size, need to re-allocate the memory
    if size(X, 1) != size(l.x, 1)
        update(l, size(X))
    end

    # Pad one at the end of the vector
    x = Array{Float64,2}(size(l.x,1), l.fan_in+1)
    x[:,1:l.fan_in] = X
    x[:,l.fan_in+1] = 1
    if x_low
        l.x = BlockFixedArray{T}(x,σ)
    else
        l.x = x
    end
    y = l.x*l.w
    if l.nonlinearity=="ReLu"
        return broadcast(max, X, 0.)
    # TODO
    else
        return l.x*l.W
    end
end

function backward(l::BlockFixedFC{T}, DLDY::Array; kwargs...) where {T<:Signed}
    @assert size(DLDY,2) == size(l.W,2)
    l.dldy = DLDY
    if length(l.base.parents)>0
        # TODO fixed
        broadcast!(>, l.base.dldx[parent_id], l.x, 0.)
        temp = BlockFixedArray{T}(l.W.arr', l.dldy.σ)
        temp = l.dldy * temp
        l.base.dldx[l.base.parents[1].base.id] = float(BlockFixedArray{T}(temp.arr[:, 1:end-1],l.dldy.σ))
    end
    l.grad = l.x*l.dldy
end

function getGradient(l::BlockFixedFC)
    Array[l.grad]
end

function getParam(l::BlockFixedFC)
    Array[l.W]
end

function setParam!(l::BlockFixedFC, theta)
    @assert size(l.W) == size(theta[1])
    l.W = theta[1]
end

function getVelocity(l::BlockFixedFC)
    return Array[l.velc]
end

# Define the Fully Connected layers
# include("LayerBase.jl")
type FullyConnected16 <: Layer
    base    :: LayerBase
    x       :: Array{Float16}
    dldy    :: Array{Float16}

    init_type :: String
    i         :: Int
    num_units :: Int
    W         :: Array{Float16}
    velc      :: Array{Float16}
    grad      :: Array{Float16}

    function FullyConnected16(prev::Layer, num_units::Int; init_type="He")
        i, o = 1, num_units
        layer = new(LayerBase(), Float16[], Float16[], init_type,
            i, o,
            convert16(randn(i+1,o)),
            convert16(zeros(i+1, o)),
            convert16(zeros(i+1, o)))
        out_size = getOutputSize(prev)
        connect(layer, [prev])
        @assert length(layer.base.parents) == 1
        init(layer, out_size)
        layer
    end

    function FullyConnected16(config, num_units::Int; init_type="He")
        i, o = 1, num_units
        layer = new(LayerBase(), Float16[], Float16[], Float16[], init_type,
            i, o, randn(i+1,o), zeros(i+1, o), zeros(i+1, o))
        @assert length(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
        init(layer, out_size)
        layer
    end

    # Create a copy of another FC layer; used for multi threading
    function FullyConnected16(main::FullyConnected16)
        layer = new(LayerBase(), Float16[], Float16[], Float16[], "",
            0, 0, randn(1,1), zeros(1, 1), zeros(1, 1))
        layer.base.y = copy(main.base.y)
        layer.base.dldx = copy(main.base.dldx)
        layer.x = copy(main.x)
        layer.dldy = copy(main.dldy)
        layer.init_type = main.init_type
        layer.i = copy(main.i)
        layer.num_units = copy(main.num_units)
        layer.W = main.W
        layer.velc = copy(main.velc)
        layer.grad = copy(main.grad)
        layer
    end
end

function convert16(a)
    convert(Array{Float16},a)
end

verbose = 0

function init(l::FullyConnected16, out_size::Tuple; kwargs...)
    """
    [l]         the layer to be initialized
    [out_size]  the size of the output matrix
    """


    batch_size, input_size = out_size
    l.i = input_size

    # Get enough information, now preallocate the memory
    l.x     = Array{Float16}(batch_size, l.i + 1)
    l.base.y = Array{Float16}(batch_size, l.num_units)
    l.dldy  = Array{Float16}(batch_size, l.num_units)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(batch_size, l.i + 1)
    end
    l.velc  = convert16(zeros(l.i + 1,    l.num_units))
    l.grad  = convert16(zeros(l.i + 1,    l.num_units))

    # Pull out the output size
    fan_in, fan_out = l.i, l.num_units
    if l.init_type == "Uniform"
        local a    = sqrt(12. / (fan_in + fan_out))
        l.W = convert16(rand(fan_in+1,fan_out)* 2 * a - a)
    elseif l.init_type == "He"
        σ = sqrt(2. / fan_in)
        l.W  = rconvert16(andn(fan_in+1,fan_out) * σ)
    elseif l.init_type == "Glorot"
        σ = sqrt(2. / (fan_in+fan_out))
        l.W  = convert16(randn(fan_in+1,fan_out) * σ)
    elseif l.init_type == "Random"
        l.W = convert16(rand(fan_in+1,fan_out) - 0.5)
    end
    l.W[fan_in+1,:] = convert16(zeros(fan_out))

end

function update(l::FullyConnected16, input_size::Tuple;)
    # Reinitialize the memory due to the updated of the batch_size
    # Couldn't change the input and output size, only the bath size
    # the outter dimension must be the same, so that we don't need
    # to reinitialize the weights and bias
    @assert length(input_size) == 2 && size(l.x, 2) == size(l.x, 2)
    batch_size = input_size[1]
    l.x      = Array{Float16}(batch_size, l.i + 1)
    l.base.y = Array{Float16}(batch_size, l.num_units)
    l.dldy   = Array{Float16}(batch_size, l.num_units)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id]  = Array{Float64}(batch_size, l.i + 1)
    end
    # println("FullyConnected16 update:\n\tInput:$(size(l.x))\n\tOutput:$(size(l.y))")
end

function forward(l::FullyConnected16, X::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    # X      : NxI matrix, N is the mini batch size, I is the input size
    # Output : NxO matrix
    @assert size(X)[2] == l.i

    # update the batch_size, need to re-allocate the memory
    if size(X, 1) != size(l.x, 1)
        update(l, size(X))
    end

    # Pad one at the end of the vector
    l.x[:,1:l.i] = X
    l.x[:,l.i+1] = 1

    l.x=convert16(l.x)

    # Multiplication inplaces
    l.base.y = convert(Array{Float64}, l.x*l.W)
    return l.base.y
end

function backward(l::FullyConnected16, DLDY::Array; kwargs...)
    @assert size(DLDY,2) == size(l.W,2)
    l.dldy = convert16(DLDY)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = convert(Array{Float64},l.dldy*l.W'[:, 1:end-1])
    end
    At_mul_B!(l.grad, l.x, l.dldy)
end

function getGradient(l::FullyConnected16)
  return Array[l.grad]
end

function getParam(l::FullyConnected16)
    return Array[l.W]
end

function setParam!(l::FullyConnected16, theta)
    @assert size(l.W) == size(theta[1])
    # broadcast!(-, l.velc, theta, l.W)
    l.velc = theta[1] - l.W
    # l.W = theta[1]
    copy!(l.W, theta[1])
end

function getVelocity(l::FullyConnected16)
    return Array[l.velc]
end

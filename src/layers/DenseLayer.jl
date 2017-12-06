# Define the Fully Connected layers
# include("LayerBase.jl")
type DenseLayer <: Layer
    base    :: LayerBase
    x       :: Array{Float64}
    dldy    :: Array{Float64}

    init_type :: String
    i         :: Int
    num_units :: Int
    W         :: Array{Float64}
    velc      :: Array{Float64}
    grad      :: Array{Float64}

    function DenseLayer(prev::Layer, num_units::Int; init_type="Uniform")
        i, o = 1, num_units
        layer = new(LayerBase(), Float64[], Float64[], init_type,
            i, o, randn(i+1,o), zeros(i+1, o), zeros(i+1, o))
        out_size = getOutputSize(prev)
        connect(layer, [prev])
        @assert length(layer.base.parents) == 1
        init(layer, out_size)
        layer
    end

    function DenseLayer(config::Dict{String,Any}, num_units::Int; init_type="Uniform")
        i, o = 1, num_units
        layer = new(LayerBase(), Float64[], Float64[], init_type,
            i, o, randn(i+1,o), zeros(i+1, o), zeros(i+1, o))
        @assert length(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
        init(layer, out_size)
        layer
    end
end

verbose = 0

function init(l::DenseLayer, out_size::Tuple; kwargs...)
    """
    [l]         the layer to be initialized
    [out_size]  the size of the output matrix
    """


    batch_size, input_size = out_size
    l.i = input_size

    # Get enough information, now preallocate the memory
    l.x     = Array{Float64}(batch_size, l.i + 1)
    l.base.y     = Array{Float64}(batch_size, l.num_units)
    l.dldy  = Array{Float64}(batch_size, l.num_units)
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(batch_size, l.i + 1)
    l.velc  = zeros(l.i + 1,    l.num_units)
    l.grad  = zeros(l.i + 1,    l.num_units)

    # Pull out the output size
    i, o = l.i, l.num_units
    if l.init_type == "Uniform"
        local a    = sqrt(12. / (i + o))
        l.W = rand(i+1,o)* 2 * a - a
    elseif l.init_type == "Normal"
        local sigma = sqrt(2. / (i + o))
        l.W  = randn(i+1,o) * sigma
    elseif l.init_type == "Random"
        l.W = rand(i+1,o) - 0.5
    end
    l.W[i+1,:] = zeros(o)

end

function update(l::DenseLayer, input_size::Tuple;)
    # Reinitialize the memory due to the updated of the batch_size
    # Couldn't change the input and output size, only the bath size
    # the outter dimension must be the same, so that we don't need
    # to reinitialize the weights and bias
    @assert length(input_size) == 2 && size(l.x, 2) == size(l.x, 2)
    batch_size = input_size[1]
    l.x      = Array{Float64}(batch_size, l.i + 1)
    l.base.y = Array{Float64}(batch_size, l.num_units)
    l.dldy   = Array{Float64}(batch_size, l.num_units)
    l.base.dldx[l.base.parents[1].base.id]  = Array{Float64}(batch_size, l.i + 1)
    # println("DenseLayer update:\n\tInput:$(size(l.x))\n\tOutput:$(size(l.y))")
end

function forward(l::DenseLayer, X::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
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

    # Multiplication inplaces
    A_mul_B!(l.base.y, l.x, l.W)
    return l.base.y
end

function backward(l::DenseLayer, DLDY::Array; kwargs...)
    @assert size(DLDY,2) == size(l.W,2)
    l.dldy = DLDY
    parent_id = l.base.parents[1].base.id
    l.base.dldx[parent_id] = A_mul_Bt(l.dldy, l.W)[:, 1:end-1]
    At_mul_B!(l.grad, l.x, l.dldy)
end

function getGradient(l::DenseLayer)
  return Array[l.grad]
end

function getParam(l::DenseLayer)
    return Array[l.W]
end

function setParam!(l::DenseLayer, theta)
    @assert size(l.W) == size(theta[1])
    # broadcast!(-, l.velc, theta, l.W)
    l.velc = theta[1] - l.W
    l.W = theta[1]
end

function getVelocity(l::DenseLayer)
    return Array[l.velc]
end

# l = DenseLayer(800) # 800 hidden units
# X = rand(500, 784)  #input size 784, batch size 500
# Y = rand(500, 800)
#
# println("First time (compiling...)")
# @time init(l, nothing, Dict{String, Any}("batch_size" => 500, "input_size" => [784]))
# @time forward(l,X)
# @time backward(l,Y)
#
# println("Second time (profiling...)")
# @time begin
#   for i = 1:10
#     forward(l,X)
#   end
# end
# @time begin
#   for i = 1:1000
#     forward(l,X)
#   end
# end
#
# @time begin
#   for i = 1:10
#     backward(l,Y)
#   end
# end
# @time begin
#   for i = 1:1000
#     backward(l,Y)
#   end
# end

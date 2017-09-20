# Define the Fully Connected layers
# include("LayerBase.jl")
type DenseLayer <: Layer
    parents  :: Array{Layer}
    children :: Array{Layer}

    has_init  :: Bool
    init_type :: String
    i         :: Int
    num_units :: Int
    W         :: Array{Float64}
    x         :: Array{Float64}
    y         :: Array{Float64}
    dldy      :: Array{Float64}
    dldx      :: Array{Float64}
    velc      :: Array{Float64}
    grad      :: Array{Float64}

    # Minimal Initializer, needs to be initialized
    function DenseLayer(num_units::Int;init_type="Uniform")
        i, o = 1, num_units
        return new(Layer[], Layer[], false, init_type, i, o, randn(i+1,o), zeros(i), zeros(o),
                   zeros(o), zeros(i), zeros(i+1, o), zeros(i+1, o))
    end

    function DenseLayer(prev::Layer, num_units::Int, config::Dict{String,Any}; init_type="Uniform")
        i, o = 1, num_units
        layer = new(Layer[], Layer[], false, init_type, i, o, randn(i+1,o), zeros(i), zeros(o),
                   zeros(o), zeros(i), zeros(i+1, o), zeros(i+1, o))
        init(layer, prev, config)
        layer
    end
end

verbose = 0

function init(l::DenseLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    """
    [l]         the layer to be initialized
    [p]         the input layer (previous layer), assumed initialized
    [config]    the configuration of the whole network (i.e. batch, etc)
    """
	l.parents.append(p)
    if !isa(p,Void)
      p.children = [l]
    end

    if p == nothing
        # [l] is the first layer, batch_size used default network batch_size
        # and input_size should be single dimensional (i.e. vector)
        @assert length(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
    else
        out_size = getOutputSize(p)
        @assert length(out_size) == 2 # TODO: maybe a friendly error message?
    end

    batch_size, input_size = out_size
    l.i = input_size

    # Get enough information, now preallocate the memory
    l.x     = Array{Float64}(batch_size, l.i + 1)
    l.y     = Array{Float64}(batch_size, l.num_units)
    l.dldy  = Array{Float64}(batch_size, l.num_units)
    l.dldx  = Array{Float64}(batch_size, l.i + 1)
    l.velc  = zeros(l.i + 1,    l.num_units)
    l.grad  = zeros(l.i + 1,    l.num_units)

    # Pull out the output size
    i, o = l.i, l.num_units
    if l.init_type == "Uniform"
        local a    = sqrt(12. / (i + o))
        l.W = rand(i+1,o)* 2 * a - a
    elseif l.init_type == "Normal"
        local sigma = sqrt(2. / (i + o))
        l.W  = randn(i+1,o) * sqrt(sigma)
    elseif l.init_type == "Random"
        l.W = rand(i+1,o) - 0.5
    end
    l.W[i+1,:] = zeros(o)

    l.has_init = true
end

function update(l::DenseLayer, input_size::Tuple;)
    # Reinitialize the memory due to the updated of the batch_size
    # Couldn't change the input and output size, only the bath size
    # the outter dimension must be the same, so that we don't need
    # to reinitialize the weights and bias
    @assert length(input_size) == 2 && size(l.x, 2) == size(l.x, 2)
    batch_size = input_size[1]
    l.x     = Array{Float64}(batch_size, l.i + 1)
    l.y     = Array{Float64}(batch_size, l.num_units)
    l.dldy  = Array{Float64}(batch_size, l.num_units)
    l.dldx  = Array{Float64}(batch_size, l.i + 1)
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
    A_mul_B!(l.y, l.x, l.W)
    return l.y
end

function backward(l::DenseLayer, DLDY::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    @assert size(DLDY,2) == size(l.W,2)
    l.dldy = DLDY
    A_mul_Bt!(l.dldx, DLDY, l.W)
    At_mul_B!(l.grad, l.x, l.dldy) # Also compute the gradient
    return view(l.dldx, :, 1:l.i)
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

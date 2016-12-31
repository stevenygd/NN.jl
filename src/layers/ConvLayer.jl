include("LayerBase.jl")

# Assumptions:
# 1. padding doesn't work yet
# 2. stride doesn't work yet (especially for backward pass)
# 3. double check whether we need the kernel size to be odd number
type ConvLayer <: LearnableLayer
    has_init :: Bool

    # Parameters
    init_type:: String                  # Type of initialization
    pad      :: Int                     # Padding
    stride   :: Int                     # Stride
    filter   :: Int                     # Number of filters
    k_size   :: Tuple{Int, Int}         # (k_width,     k_height)
    x_size   :: Tuple{Int, Int, Int}    # (channel,     img_width,  img_height)

    # Input output place holders
    x        :: Array{Float64, 4}       # (batch_size,  channel,    width,     height)
    y        :: Array{Float64, 4}       # (batch_size,  #filter,    out_width, out_height)
    dldy     :: Array{Float64, 4}       # (batch_size,  #filter,    out_width, out_height)
    dldx     :: Array{Float64, 4}       # (batch_size,  channel,    width,     height)

    # Kernel and it's gradient & velocity
    kern     :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)
    k_grad   :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)
    k_velc   :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)

    # Biases and its gradients & velocity
    bias     :: Array{Float64, 1}       # (#filter)
    b_grad   :: Array{Float64, 1}       # (#filter)
    b_velc   :: Array{Float64, 1}       # (#filter)

    function ConvLayer(filters::Int, kernel::Tuple{Int,Int}; padding = 0, stride = 1, init="Uniform")
        @assert length(kernel) == 2 && kernel[1] % 2 == 1 &&  kernel[2] % 2 == 1
        @assert stride == 1     # doesn't support other stride yet
        @assert padding == 0    # doesn't support padding yet
        return new(false, init,
                   padding, stride, filters, kernel, (0,0,0),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1), zeros(1), zeros(1))
    end
end

function computeOutputSize(l::ConvLayer, input_size::Tuple)
    f, p, s     = l.filter, l.pad, l.stride
    x, y        = l.k_size
    b, _, w, h  = input_size
    return (b, f, convert(Int, (w+2*p-x)/s + 1), convert(Int,(h+2*p-y)/s+1))
end

function init(l::ConvLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    """
    Initialize the Convolutional layers. Preallocate all the memories.
    """
    if p == nothing
        @assert length(config["input_size"]) == 3
        batch_size = config["batch_size"]
        c, w, h    = config["input_size"]
        input_size = (batch_size, c, w, h)
    else
        input_size = getOutputSize(p)
    end
    @assert length(input_size) == 4
    output_size  = computeOutputSize(l, input_size)
    b, c, w, h   = input_size
    kw, kh       = l.k_size
    _, f, ow, oh = output_size

    # initialize input/output
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)

    # initialize weights
    f_in   = kw * kh * c    # Input number of neron: (kernel_w x kernel_h x channel)
    f_out  = f              # Output number of neron is the number of filters
    kernel_size = (f, c, kw, kh)

    if l.init_type == "Uniform"
        a = sqrt(12./(f_in + f_out))
        l.kern = rand(kernel_size) * 2 * a - a
    elseif l.init_type == "Normal"
        a = sqrt(2./(f_in + f_out))
        l.kern = randn(kernel_size) * a
    else # l.init_type == Random : )
        l.kern = rand(kernel_size) - 0.5
    end
    l.bias = zeros(l.filter)
    l.k_grad = zeros(size(l.kern))
    l.b_grad = zeros(size(l.bias))
    l.has_init = true
end

function update(l::ConvLayer, input_size::Tuple;)
    # assert: only change the batch sizes
    @assert length(input_size) == 4
    @assert input_size[2:end] == size(l.x)[2:end]

    b = input_size[1]
    output_size = size(l.y)
    output_size = (b, output_size[2], output_size[3], output_size[4])

    # Relinitialize input and output
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
end

function inner_conv2(x::Array{Float64,2}, k::Array{Float64,2})
    """
    2D convolution, inner convolution where the kernel only slides inside inside
    the image; only applicable for the 2-D images and 2-D kernels
    """
    kw, kh = size(k)
    return conv2(x,k)[kw:end-kw+1, kh:end-kh+1]
end

function outter_conv2(x::Array{Float64,2}, k::Array{Float64,2})
    """
    2D convolution, inner convolution where the kernel will slide through every
    position as long as there is intersection with the image; same as native [conv2]
    """
    return conv2(x,k)
end

function flip(x::Union{SubArray{Float64,4},Array{Float64,4}})
    return x[:,:,end:-1:1, end:-1:1]
end

function forward(l::ConvLayer, x::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    if size(l.x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    batch_size, img_depth = size(l.x, 1), size(l.x, 2)
    scale!(l.y, 0.)
    for b = 1:batch_size
    for c = 1:img_depth
    for f = 1:l.filter
        # TODO, need to test whether the in place function works this way
        l.y[b,f,:,:] += inner_conv2(l.x[b,c,:,:], l.kern[f,c,:,:])
    end; end; end
    return l.y
end

function backward(l::ConvLayer, dldy::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    l.dldy = dldy
    scale!(l.dldx, 0.)
    flipped = flip(l.kern)
    batch_size, depth = size(l.x,1), size(l.x, 2)
    for b=1:batch_size
    for f=1:l.filter
    for c=1:depth
        l.dldx[b,c,:,:] += outter_conv2(l.dldy[b,f,:,:], flipped[f,c,:,:])
    end; end; end
    return l.dldx
end

function getGradient(l::ConvLayer)
    flipped = flip(l.dldy)
    batch_size, depth = size(l.x,1), size(l.x, 2)
    scale!(l.k_grad,0.)
    tmp = Array{Float64}(l.k_size)
    for b=1:batch_size
    for f=1:l.filter
    for c=1:depth
        l.k_grad[f,c,:,:] += inner_conv2(l.x[b,c,:,:], l.dldy[b,f,:,:])
    end; end; end
    l.b_grad = sum(sum(sum(l.dldy, 4), 3), 1)[1,:,1,1]
    return (l.k_grad, l.b_grad)
end

function getParam(l::ConvLayer)
    return (l.kern, l.bias)
end

function setParam!(l::ConvLayer, theta::Array{Float64})
    new_kern, new_bias = theta

    l.k_velc = new_kern - l.kern
    l.kern   = new_kern

    l.b_velc = new_bias - l.bias
    l.bias   = new_bias
end

function getVelocity(l::ConvLayer)
    return (l.k_velc, l.b_velc)
end

l = ConvLayer(64,(3,3))
X = rand(64, 3, 30, 30)
Y = rand(64, 64, 28, 28)

println("First time (compiling...)")
@time init(l, nothing, Dict{String, Any}("batch_size" => 64, "input_size" => (3,30,30)))
@time forward(l,X)
@time backward(l,Y)
@time getGradient(l)

# Assumptions:
# 1. padding doesn't work yet
# 2. stride doesn't work yet (especially for backward pass)
# 3. double check whether we need the kernel size to be odd number
type Conv <: LearnableLayer
    base     :: LayerBase

    # Parameters
    init_type:: String                  # Type of initialization
    pad      :: Int                     # Padding
    stride   :: Int                     # Stride
    filter   :: Int                     # Number of filters
    k_size   :: Tuple{Int, Int}         # (k_width,     k_height)
    x_size   :: Tuple{Int, Int, Int}    # (channel,     img_width,  img_height)

    # Input output place holders
    x        :: Array{Float64, 4}       # (batch_size,  channel,    width,     height)
    dldy     :: Array{Float64, 4}       # (batch_size,  #filter,    out_width, out_height)
    dldx_cache :: Array{Float64, 4}

    # Kernel and it's gradient & velocity
    kern     :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)
    k_grad   :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)
    k_velc   :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)

    # Biases and its gradients & velocity
    bias     :: Array{Float64, 1}       # (#filter)
    b_grad   :: Array{Float64, 1}       # (#filter)
    b_velc   :: Array{Float64, 1}       # (#filter)

    function Conv(prev::Layer, filters::Int, kernel::Tuple{Int,Int}; padding = 0, stride = 1, init_type="Uniform")
        # @assert length(kernel) == 2 && kernel[1] % 2 == 1 &&  kernel[2] % 2 == 1
        @assert stride == 1     # doesn't support other stride yet
        @assert padding == 0    # doesn't support padding yet
        layer = new(LayerBase(), init_type,
                   padding, stride, filters, kernel, (0,0,0),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1), zeros(1), zeros(1))
        connect(layer, [prev])
        init(layer, getOutputSize(prev))
        layer
    end

    function Conv(config::Dict{String,Any}, filters::Int, kernel::Tuple{Int,Int}; padding = 0, stride = 1, init_type="Normal")
        @assert stride == 1
        @assert padding == 0
        layer = new(LayerBase(), init_type,
                   padding, stride, filters, kernel, (0,0,0),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1), zeros(1), zeros(1))
        input_size = (config["batch_size"]..., config["input_size"]...)
        init(layer, input_size)
        layer
    end
end

function computeOutputSize(l::Conv, input_size::Tuple)
    f, p, s     = l.filter, l.pad, l.stride
    x, y        = l.k_size
    b, _, w, h  = input_size
    return (b, f, convert(Int, (w+2*p-x)/s + 1), convert(Int,(h+2*p-y)/s+1))
end

function init(l::Conv, input_size::Tuple; kwargs...)
    """
    Initialize the Convolutional layers. Preallocate all the memories.
    """
    print(input_size)
    @assert length(input_size) == 4
    output_size  = computeOutputSize(l, input_size)
    b, c, w, h   = input_size
    kw, kh       = l.k_size
    _, f, ow, oh = output_size

    # initialize input/output
    l.x    = Array{Float64}(input_size)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    end
    l.dldx_cache = Array{Float64}(input_size)
    l.base.y    = Array{Float64}(output_size)
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
    l.k_velc = zeros(size(l.kern))
    l.b_velc = zeros(size(l.bias))
end

function update(l::Conv, input_size::Tuple;)
    # assert: only change the batch sizes
    @assert length(input_size) == 4
    @assert input_size[2:end] == size(l.x)[2:end]

    b = input_size[1]
    output_size = size(l.base.y)
    output_size = (b, output_size[2], output_size[3], output_size[4])

    # Relinitialize input and output
    l.x    = Array{Float64}(input_size)
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    end
    l.base.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)

    println("Conv update shape:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

tensor2 = Union{SubArray{Float64,2},Array{Float64,2}}
tensor4 = Union{SubArray{Float64,4},Array{Float64,4}}

function inner_conv2(x::tensor2, k::tensor2)
    """
    2D convolution, inner convolution where the kernel only slides inside inside
    the image; only applicable for the 2-D images and 2-D kernels
    """
    kw, kh = size(k)
    return conv2(x,k)[kw:end-kw+1, kh:end-kh+1]
end

function outter_conv2(x::tensor2, k::tensor2)
    """
    2D convolution, inner convolution where the kernel will slide through every
    position as long as there is intersection with the image; same as native [conv2]
    """
    return conv2(x,k)
end

function flip(x::tensor4)
    return x[:,:,end:-1:1, end:-1:1]
end

function forward(l::Conv, x::tensor4; kwargs...)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    batch_size, img_depth = size(l.x, 1), size(l.x, 2)
    scale!(l.base.y, 0.)
    for b = 1:batch_size
    for c = 1:img_depth
    for f = 1:l.filter
        l.base.y[b,f,:,:] += inner_conv2(view(l.x,b,c,:,:), view(l.kern,f,c,:,:))
    end; end; end
    return l.base.y
end

function backward(l::Conv, dldy::tensor4; kwargs...)
    l.dldy = dldy
    l.dldx_cache = zeros(size(l.x))
    flipped = flip(l.kern)
    batch_size, depth = size(l.x,1), size(l.x, 2)
    for b=1:batch_size
    for f=1:l.filter
    for c=1:depth
        l.dldx_cache[b,c,:,:] += outter_conv2(view(l.dldy,b,f,:,:), view(flipped,f,c,:,:))
    end; end; end
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = l.dldx_cache
    end
    l.dldx_cache
end

function getGradient(l::Conv)
    flipped = flip(l.dldy)
    batch_size, depth = size(l.x,1), size(l.x, 2)
    scale!(l.k_grad,0.)
    for b=1:batch_size
    for f=1:l.filter
    for c=1:depth
        l.k_grad[f,c,:,:] += inner_conv2(view(l.x,b,c,:,:), view(l.dldy,b,f,:,:))
    end; end; end
    l.b_grad = sum(sum(sum(l.dldy, 4), 3), 1)[1,:,1,1]

    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.k_grad
    ret[:,end,1,1]     = l.b_grad
    return ret
end

function getParam(l::Conv)
    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.kern
    ret[:,end,1,1]     = l.bias
    return ret
end

function setParam!(l::Conv, theta::Array{Float64})
    # convention: ret[:,end,1,1] is the gradient for bias

    new_kern, new_bias = theta[:,1:end-1,:,:], theta[:,end,1,1]

    l.k_velc = new_kern - l.kern
    l.kern   = new_kern

    l.b_velc = new_bias - l.bias
    l.bias   = new_bias
end

function getVelocity(l::Conv)
    # return (l.k_velc, l.b_velc)
    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.k_velc
    ret[:,end,1,1]     = l.b_velc
    return ret
end

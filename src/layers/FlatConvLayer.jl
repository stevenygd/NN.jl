include("LayerBase.jl")

# Assumptions:
# 1. padding doesn't work yet
# 2. stride doesn't work yet (especially for backward pass)
# 3. double check whether we need the kernel size to be odd number
type FlatConvLayer <: LearnableLayer

    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init  :: Bool
    id        :: Base.Random.UUID

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

    function FlatConvLayer(filters::Int, kernel::Tuple{Int,Int}; padding = 0, stride = 1, init="Uniform")
        @assert length(kernel) == 2 && kernel[1] % 2 == 1 &&  kernel[2] % 2 == 1
        @assert stride == 1     # doesn't support other stride yet
        @assert padding == 0    # doesn't support padding yet
        return new(Layer[], Layer[], false, Base.Random.uuid4(), init,
                   padding, stride, filters, kernel, (0,0,0),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1), zeros(1), zeros(1))
    end

    function FlatConvLayer(prev::Layer, filters::Int, kernel::Tuple{Int,Int}; config::Union{Dict{String,Any},Void}=nothing, padding = 0, stride = 1, init="Uniform")
        @assert length(kernel) == 2 && kernel[1] % 2 == 1 &&  kernel[2] % 2 == 1
        @assert stride == 1     # doesn't support other stride yet
        @assert padding == 0    # doesn't support padding yet
        layer = new(Layer[], Layer[], false, Base.Random.uuid4(), init,
                   padding, stride, filters, kernel, (0,0,0),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1), zeros(1), zeros(1))
        init(layer, prev, config; kwargs...)
        layer
    end
end

function computeOutputSize(l::FlatConvLayer, input_size::Tuple)
    f, p, s     = l.filter, l.pad, l.stride
    x, y        = l.k_size
    b, _, w, h  = input_size
    return (b, f, convert(Int, (w+2*p-x)/s + 1), convert(Int,(h+2*p-y)/s+1))
end

function init(l::FlatConvLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)

    if !isa(p,Void)
        l.parents = [p]
        push!(p.children, l)
    end

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
    l.k_velc = zeros(size(l.kern))
    l.b_velc = zeros(size(l.bias))
    l.has_init = true
end

function update(l::FlatConvLayer, input_size::Tuple;)
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

    println("ConvLayer update shape:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

tensor2 = Union{SubArray{Float64,2},Array{Float64,2}}
tensor4 = Union{SubArray{Float64,4},Array{Float64,4}}

function flip(x::tensor4)
    return x[:,:,end:-1:1, end:-1:1]
end

function conv4d(x::tensor4, kern::tensor4, bias::Array{Float64, 1}, inner::Bool)
    b, c,  w,   h   = size(x)
    f, c2, k_w, k_h = size(kern)
    o_w, o_h       = w+k_w-1, h+k_h-1
    @assert c2 == c

    # Initialize all the memories
    forward_x = Array{Float64}(c, w,   (b-1)*(h+k_h-1)+h)
    tmp_coved = Array{Float64}(c, o_w, o_h*b)
    tmp_sumed = Array{Float64}(f, o_w, o_h*b)
    tmp_out   = Array{Float64}(b, f, o_w, o_h)

    # Ship the data
    fill!(forward_x, 0.)
    for c_i = 1:c
    for b_i = 1:b
        start_idx = (b_i-1)*(h+k_h-1)+1
        end_idx   = start_idx + h - 1
        forward_x[c_i, :, start_idx:end_idx] = x[b_i, c_i, :, :]
    end
    end

    # apply each filter channel wise
    fill!(tmp_sumed, 0.)
    for f_i = 1:f
        fill!(tmp_coved, 0.)
        for c_i = 1:c
            tmp_coved[c_i, :, :] = conv2(forward_x[c_i, :, :], kern[f_i, c_i, :, :])
        end
        tmp_sumed[f_i, :, :] = sum(tmp_coved, 1)
        broadcast!(+, tmp_sumed[f_i:f_i, :, :], tmp_sumed[f_i:f_i, :, :], bias[f_i])
    end


    # reshape to the desirable shape
    for b_i = 1:b
        tmp_out[b_i,:,:,:] = tmp_sumed[:,:,(o_h*(b_i-1)+1):(o_h*b_i)]
    end

    if inner
        return tmp_out[:,:,k_w:end-k_w+1, k_h:end-k_h+1]
    else
        return tmp_out
    end
end

function forward(l::FlatConvLayer, x::tensor4; kwargs...)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    l.y = conv4d(l.x, l.kern, l.bias, true) # inner convolution
    return l.y
end

function backward(l::FlatConvLayer, dldy::tensor4; kwargs...)
    l.dldy = dldy
    flipped = permutedims(flip(l.kern), [2,1,3,4])
    f = size(flipped,1)
    l.dldx  = conv4d(l.dldy, flipped, zeros(f), false) # outter convolution
    return l.dldx
end

function getGradient(l::FlatConvLayer)
    flipped  = permutedims(flip(l.dldy), [2,1,3,4])
    f = size(flipped,1)
    flipped_x = permutedims(l.x, [2,1,3,4])
    l.k_grad =  permutedims(conv4d(flipped_x, flipped, zeros(f), true), [2,1,3,4]) # outter convolution
    batch_size, depth = size(l.x,1), size(l.x, 2)
    l.b_grad = sum(sum(sum(l.dldy, 4), 3), 1)[1,:,1,1]

    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.k_grad
    ret[:,end,1,1]     = l.b_grad
    return ret
end

function getParam(l::FlatConvLayer)
    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.kern
    ret[:,end,1,1]     = l.bias
    return ret
end

function setParam!(l::FlatConvLayer, theta::Array{Float64})
    # convention: ret[:,end,1,1] is the gradient for bias

    new_kern, new_bias = theta[:,1:end-1,:,:], theta[:,end,1,1]

    l.k_velc = new_kern - l.kern
    l.kern   = new_kern

    l.b_velc = new_bias - l.bias
    l.bias   = new_bias
end

function getVelocity(l::FlatConvLayer)
    # return (l.k_velc, l.b_velc)
    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.k_velc
    ret[:,end,1,1]     = l.b_velc
    return ret
end

# include("ConvLayer.jl")
# using Base.Test
# l = FlatConvLayer(16,(3,3))
# l2= ConvLayer(16,(3,3))
# X = rand(32, 3, 10, 10)
# Y = rand(32, 16, 8, 8)
#
# println("First time (compiling...)")
# init(l, nothing, Dict{String, Any}("batch_size" => 32, "input_size" => (3,10,10)))
# init(l2, nothing, Dict{String, Any}("batch_size" => 32, "input_size" => (3,10,10)))
# l2.kern = l.kern
# @time y1 = forward(l,X)
# @time y2 = forward(l2,X)
# @test_approx_eq y1 y2
#
# @time y1 = backward(l,Y)
# @time y2 = backward(l2,Y)
# @test_approx_eq y1 y2
#
# @time y1 = getGradient(l)
# @time y2 = getGradient(l2)
# @test_approx_eq y1 y2
#
# println("Second time (after compilation...)")
# @time forward(l,X)
# @time backward(l,Y)
# @time getGradient(l)

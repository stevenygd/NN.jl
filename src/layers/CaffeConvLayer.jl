include("LayerBase.jl")

# Assumptions:
# 1. padding doesn't work yet
# 2. stride doesn't work yet (especially for backward pass)
# 3. double check whether we need the kernel size to be odd number
type CaffeConvLayer <: LearnableLayer
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

    function CaffeConvLayer(filters::Int, kernel::Tuple{Int,Int}; padding = 0, stride = 1, init="Uniform")
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

function computeOutputSize(l::CaffeConvLayer, input_size::Tuple)
    f, p, s     = l.filter, l.pad, l.stride
    x, y        = l.k_size
    b, _, w, h  = input_size
    return (b, f, convert(Int, (w+2*p-x)/s + 1), convert(Int,(h+2*p-y)/s+1))
end

function init(l::CaffeConvLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
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

function update(l::CaffeConvLayer, input_size::Tuple;)
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
tensor3 = Union{SubArray{Float64,3},Array{Float64,3}}
tensor4 = Union{SubArray{Float64,4},Array{Float64,4}}

function flip(x::tensor4)
    return x[:,:,end:-1:1, end:-1:1]
end

function caffe_conv4d_masking(x::tensor4, kern::tensor4, bias::Array{Float64, 1}, inner::Bool; stride=1)
    b, c,  w, h     = size(x)
    f, c2, k_w, k_h = size(kern)
    o_w, o_h  = w+k_w-1, h+k_h-1
    @assert c == c2

    # Initialize memory
    mask     = Array{CartesianIndex{3}}(o_w*o_h,   c*k_w*k_h)
    m_conved = zeros(o_w*o_h,   f)
    m_transp = zeros(f, o_w, o_h)
    output   = zeros(b, f, o_w, o_h)

    row = 1
    for nx = 1:o_w
    for ny = 1:o_h
        # Compute starting points
        ix = 1 + (nx - 1) * stride
        iy = 1 + (ny - 1) * stride
        c_range = CartesianRange(
            CartesianIndex(1, ix      , iy),
            CartesianIndex(c, ix+k_w-1, iy+k_h-1)
        )
        col = 1
        for idx in c_range
            mask[row, col] = idx
            col+=1
        end
        row+=1
    end
    end

    # flatten a 3D-kernel
    m_ker  = zeros(c*k_w*k_h, f)
    for col = 1:f
        m_ker[:,col] = reshape(kern[col,:,:,:], 1,c*k_w*k_h)
    end

    # Padded images
    x_p = zeros(b, c, w+2*k_w-2, h+2*k_h-2)
    x_p[:,:,k_w:w+k_w-1,k_h:h+k_h-1] = x

    # For each batch, fill m_img, and make computation
    for nb = 1:b
        A_mul_B!(m_conved, x_p[nb,:,:,:][mask], m_ker)
        m_reshpe = reshape(m_conved, o_w, o_h, f)
        permutedims!(m_transp, m_reshpe, [3,1,2])
        output[nb,:,:,:] = m_transp
    end

    if inner
        return output[:,:,k_w:end-k_w+1, k_h:end-k_h+1]
    else
        return output
    end
end

function caffe_conv4d(x::tensor4, kern::tensor4, bias::Array{Float64, 1}, inner::Bool;stride=1)
    b, c,  w,   h   = size(x)
    f, c2, k_w, k_h = size(kern)
    o_w, o_h       = w+k_w-1, h+k_h-1
    @assert c2 == c

    # Initialize memory
    m_img    = zeros(o_w*o_h,   c*k_w*k_h)
    m_ker    = zeros(c*k_w*k_h, f)
    m_conved = zeros(o_w*o_h,   f)
    m_transp = zeros(f, o_w, o_h)
    output   = zeros(b, f, o_w, o_h)

    # Fill m_ker
    for col = 1:f
        m_ker[:,col] = reshape(kern[col,:,:,:], 1,c*k_w*k_h) # flatten a 3D-kernel
    end

    # Padded images
    x_p = zeros(b, c, w+2*k_w-2, h+2*k_h-2)
    x_p[:,:,k_w:w+k_w-1,k_h:h+k_h-1] = x

    # For each batch, fill m_img, and make computation
    for nb = 1:b
        row = 1
        fill!(m_img, 0.)
        for nx = 1:o_w
        for ny = 1:o_h
            # Compute starting points
            ix = 1 + (nx - 1) * stride
            iy = 1 + (ny - 1) * stride
            col = 1
            for ic = 1:c
            for iw = ix:ix+k_w-1
            for ih = iy:iy+k_h-1
              m_img[row, col] = x_p[nb,ic,iw,ih]
              col += 1
            end
            row += 1
        end
        end
        A_mul_B!(m_conved, m_img, m_ker)
        m_reshpe = reshape(m_conved, o_w, o_h, f)
        permutedims!(m_transp, m_reshpe, [3,1,2])
        output[nb,:,:,:] = m_transp
    end

    if inner
        return output[:,:,k_w:end-k_w+1, k_h:end-k_h+1]
    else
        return output
    end
end

function forward(l::CaffeConvLayer, x::tensor4; kwargs...)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    l.y = caffe_conv4d(l.x, l.kern, l.bias, true) # inner convolution
    return l.y
end

function backward(l::CaffeConvLayer, dldy::tensor4; kwargs...)
    l.dldy = dldy
    flipped = permutedims(flip(l.kern), [2,1,3,4])
    f = size(flipped,1)
    l.dldx  = caffe_conv4d(l.dldy, flipped, zeros(f), false) # outter convolution
    return l.dldx
end

function getGradient(l::CaffeConvLayer)
    flipped  = permutedims(flip(l.dldy), [2,1,3,4])
    f = size(flipped,1)
    flipped_x = permutedims(l.x, [2,1,3,4])
    l.k_grad =  permutedims(caffe_conv4d(flipped_x, flipped, zeros(f), true), [2,1,3,4]) # outter convolution
    batch_size, depth = size(l.x,1), size(l.x, 2)
    l.b_grad = sum(sum(sum(l.dldy, 4), 3), 1)[1,:,1,1]

    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.k_grad
    ret[:,end,1,1]     = l.b_grad
    return ret
end

function getParam(l::CaffeConvLayer)
    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.kern
    ret[:,end,1,1]     = l.bias
    return ret
end

function setParam!(l::CaffeConvLayer, theta::Array{Float64})
    # convention: ret[:,end,1,1] is the gradient for bias

    new_kern, new_bias = theta[:,1:end-1,:,:], theta[:,end,1,1]

    l.k_velc = new_kern - l.kern
    l.kern   = new_kern

    l.b_velc = new_bias - l.bias
    l.bias   = new_bias
end

function getVelocity(l::CaffeConvLayer)
    # return (l.k_velc, l.b_velc)
    # convention: ret[:,end,1,1] is the gradient for bias
    ret = zeros(Float64, l.filter, size(l.x, 2)+1, l.k_size[1], l.k_size[2])
    ret[:,1:end-1,:,:] = l.k_velc
    ret[:,end,1,1]     = l.b_velc
    return ret
end

bsize= 500
l = CaffeConvLayer(32,(3,3))
X = rand(bsize, 3, 27, 27)
Y = rand(bsize, 32, 25, 25)

println("First time (compiling...)")
init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => (3, 27, 27)))
@time y1 = forward(l,X)

@time y1 = backward(l,Y)

@time y1 = getGradient(l)

println("Second time (after compilation) CaffeConvLayer")
X = rand(bsize, 3, 10, 10)
Y = rand(bsize, 16, 8, 8)
@time begin
    forward(l,X)
end
@time begin
    backward(l,Y)
end
@time begin
    getGradient(l)
end

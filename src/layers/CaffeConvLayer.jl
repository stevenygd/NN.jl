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
    x        :: Array{Float64, 4}       # (width,       height,     channel,   batch_size)
    y        :: Array{Float64, 4}       # (out_width,   out_height, #filter,   batch_size)
    dldy     :: Array{Float64, 4}       # (width,       height,     channel,   batch_size)
    dldx     :: Array{Float64, 4}       # (out_width,   out_height, #filter,   batch_size)

    # Kernel and it's gradient & velocity
    kern     :: Array{Float64, 4}       # (k_width,   k_height,     channel,   #filter)
    k_grad   :: Array{Float64, 4}       # (k_width,   k_height,     channel,   #filter)
    k_velc   :: Array{Float64, 4}       # (k_width,   k_height,     channel,   #filter)
    k_grad_tmp :: Array{Float64, 4}     # (k_width,   k_height,     #filter,   channel)

    # Biases and its gradients & velocity
    bias     :: Array{Float64, 1}       # (#filter)
    b_grad   :: Array{Float64, 1}       # (#filter)
    b_velc   :: Array{Float64, 1}       # (#filter)

    # Temps for computations
    tmps_forward  :: Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    tmps_backward :: Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}
    tmps_gradient :: Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}

    function CaffeConvLayer(filters::Int, kernel::Tuple{Int,Int}; padding = 0, stride = 1, init="Uniform")
        @assert length(kernel) == 2 && kernel[1] % 2 == 1 &&  kernel[2] % 2 == 1
        @assert stride == 1     # doesn't support other stride yet
        @assert padding == 0    # doesn't support padding yet
        return new(false, init,
                   padding, stride, filters, kernel, (0,0,0),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1), zeros(1), zeros(1),
                   (zeros(1,1), zeros(1,1), zeros(1,1)), # tmps_forward
                   (zeros(1,1), zeros(1,1), zeros(1,1)), # tmps_backward
                   (zeros(1,1), zeros(1,1), zeros(1,1))) # tmps_gradient
    end
end

function computeOutputSize(l::CaffeConvLayer, input_size::Tuple)
    f, p, s     = l.filter, l.pad, l.stride
    x, y        = l.k_size
    w, h, _, b  = input_size
    return (convert(Int, (w+2*p-x)/s + 1), convert(Int,(h+2*p-y)/s+1), f, b)
end

function init(l::CaffeConvLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    """
    Initialize the Convolutional layers. Preallocate all the memories.
    """
    if p == nothing
        @assert length(config["input_size"]) == 3
        batch_size = config["batch_size"]
        w, h, c    = config["input_size"]
        input_size = (w, h, c, batch_size)
    else
        input_size = getOutputSize(p)
    end
    @assert length(input_size) == 4
    output_size  = computeOutputSize(l, input_size)
    w, h, c,  b  = input_size
    kw, kh       = l.k_size
    ow, oh, f, _ = output_size

    # initialize input/output
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)

    # initialize weights
    f_in   = kw * kh * c    # Input number of neron: (kernel_w x kernel_h x channel)
    f_out  = f              # Output number of neron is the number of filters
    kernel_size = (kw, kh, c, f)

    if l.init_type == "Uniform"
        a = sqrt(12./(f_in + f_out))
        l.kern = rand(kernel_size) * 2 * a - a
    elseif l.init_type == "Normal"
        a = sqrt(2./(f_in + f_out))
        l.kern = randn(kernel_size) * a
    else # l.init_type == Random : )
        l.kern = rand(kernel_size) - 0.5
    end
    l.bias   = zeros(l.filter)
    l.k_grad = zeros(size(l.kern))
    l.k_grad_tmp = zeros(size(l.kern,1),size(l.kern,2),size(l.kern,4),size(l.kern,3))
    l.b_grad = zeros(size(l.bias))
    l.k_velc = zeros(size(l.kern))
    l.b_velc = zeros(size(l.bias))

    # initializing the temp memories
    # TODO: output_size here should be that of the full outer-convolution
    l.tmps_forward = (
        # output_size should be that of the outter convolution,
        # Although only the inner convolution is used
        Array{Float64}(ow * oh,     c * kw * kh),
        Array{Float64}(kw * kh * c, f),
        Array{Float64}(ow * oh,     f)
    )

    l.tmps_backward = (
        Array{Float64}(w * h,       f * kw * kh),
        Array{Float64}(kw * kh * f, c),
        Array{Float64}(w * h,       c)
    )

    l.tmps_gradient = (
        Array{Float64}(kw * kh,     ow * oh * b),
        Array{Float64}(ow * oh * b, f),
        Array{Float64}(kw * kh,     f)
    )

    l.has_init = true
end

# Codes from Moncha.jl
function im2col_impl{T}(img::Array{T}, col::Array{T},
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})

  width, height, channels = size(img,1), size(img,2), size(img,3)
  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
  width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
  channels_col = channels * kernel_h * kernel_w

  for c = 0:channels_col-1
    w_offset = c % kernel_w
    h_offset = div(c, kernel_w) % kernel_h
    c_im = div(c, kernel_h * kernel_w) # channel
    for h = 0:height_col-1
      for w = 0:width_col-1
        h_pad = h*stride_h - pad_h + h_offset
        w_pad = w*stride_w - pad_w + w_offset
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
            # col[w, h, c] = img[w_pad, h_pad, c_im]
            @inbounds col[1 + (c*height_col+h) * width_col + w] =
                img[1 + (c_im * height + h_pad) * width + w_pad]
        else
          # col[w, h, c] = 0
          # assume size(col) = (width_col * height_col, channels_col)
          @inbounds col[1 + (c*height_col+h) * width_col + w] = 0
        end
      end
    end
  end
end

function col2im_impl{T}(col::Array{T}, img::Array{T}, input_size::NTuple{3,Int},
    kernel::NTuple{2,Int}, pad::NTuple{2,Int}, stride::NTuple{2,Int})
   # = size(img,1), size(img,2), size(img,3)
  width, height, channels = input_size
  kernel_w, kernel_h = kernel
  pad_w, pad_h = pad
  stride_w, stride_h = stride

  height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
  width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
  channels_col = channels * kernel_h * kernel_w

  fill!(img, 0)
  for c = 0:channels_col-1
    w_offset = c % kernel_w
    h_offset = div(c, kernel_w) % kernel_h
    c_im = div(c, kernel_w * kernel_h)
    for h = 0:height_col-1
      for w = 0:width_col-1
        h_pad = h * stride_h - pad_h + h_offset
        w_pad = w * stride_w - pad_w + w_offset
        if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
        #   @inbounds img[1 + (c_im * height + h_pad) * width + w_pad] +=
        #       col[1 + (c * height_col + h) * width_col + w]
        # Not inbounds for checking the indexing
           img[1 + (c_im * height + h_pad) * width + w_pad] +=
            col[1 + (c * height_col + h) * width_col + w]

        end
      end
    end
  end
end

function update(l::CaffeConvLayer, input_size::Tuple;)
    # assert: only change the batch sizes
    @assert length(input_size) == 4
    @assert input_size[1:2] == size(l.x)[1:2]

    b = input_size[1]
    output_size = size(l.y)
    output_size = (b, output_size[2], output_size[3], output_size[4])
    ow, oh, f, _ = output_size
    w,  h,  c, _ = input_size
    kw, kh, _, _ = size(l.kern)

    # Relinitialize input and output
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)

    # TODO: output_size here should be that of the full outer-convolution
    l.tmps_forward = (
        # output_size should be that of the outter convolution,
        # Although only the inner convolution is used
        Array{Float64}(ow * oh,     c * kw * kh),
        Array{Float64}(kw * kh * c, f),
        Array{Float64}(ow * oh,     f)
    )

    l.tmps_backward = (
        Array{Float64}(w * h,       f * kw * kh),
        Array{Float64}(kw * kh * f, c),
        Array{Float64}(w * h,       c)
    )

    l.tmps_gradient = (
        Array{Float64}(kw * kh,     ow * oh * b),
        Array{Float64}(ow * oh * b, f),
        Array{Float64}(kw * kh,     f)
    )

    println("ConvLayer update shape:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

tensor2 = Union{SubArray{Float64,2},Array{Float64,2}}
tensor3 = Union{SubArray{Float64,3},Array{Float64,3}}
tensor4 = Union{SubArray{Float64,4},Array{Float64,4}}

function flip(x::tensor4)
    # return x[:,:,end:-1:1, end:-1:1]
    return x[end:-1:1, end:-1:1,:,:]
end

function caffe_conv4d!(output::tensor4, tmps::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}},
         x::tensor4, kern::tensor4, bias::Array{Float64, 1}, inner::Bool;
         stride=1)
    """
    [output] size (o_w, o_h, #f, b)
    [x]      size (w,   h,   c,  b)
    [tmps]   contains 3 different arrays (m_img, m_ker, m_conved), each of the following size:
             [m_img]    (o_w*o_h,   c*k_w*k_h)
             [m_ker]    (c*k_w*k_h, f)
             [m_conved] (o_w*o_h,   f)
    [kern]   size : (k_width,   k_height, channel,  #filter)
    """
    _, t_a, a_a, _, _ = @timed begin
        w,   h,   c,  b = size(x)
        k_w, k_h, c2, f = size(kern)
        o_w, o_h        = inner?(w-k_w+1):(w+k_w-1), inner?(h-k_h+1):(h+k_h-1)
        kernel          = (k_w, k_h)
        @assert c2 == c

        m_img, m_ker, m_conved = tmps
        fill!(m_img, 0.)
        fill!(m_ker, 0.)
        fill!(m_conved, 0.)
        fill!(output, 0.)
    end

    # Fill m_ker
    _, t_rshp, a_rshp, _, _ = @timed for col = 1:f
        m_ker[:,col] = reshape(kern[:,:,:,col], 1,c*k_w*k_h) # flatten a 3D-kernel
    end


    t_1, t_2, t_3 = 0., 0., 0.
    a_1, a_2, a_3 = 0., 0., 0.
    for nb = 1:b
        if inner
            _, t1_curr, a1_curr, _, _ = @timed im2col_impl(x[:,:,:,nb], m_img, kernel, (0,0), (stride,stride))
        else # outter convolution, add padding
            _, t1_curr, a1_curr, _, _ = @timed im2col_impl(x[:,:,:,nb], m_img, kernel, (k_w,k_h), (stride,stride))
        end

        _, t2_curr, a2_curr, _, _ = @timed A_mul_B!(m_conved, m_img, m_ker)
        _, t3_curr, a3_curr, _, _ = @timed begin
            # TODO: this could be wrong
            m_transp = reshape(m_conved, o_w, o_h, f)
            # if inner
            #     output[:,:,:,nb] = m_transp[k_w:end-k_w+1, k_h:end-k_h+1,:]
            # else
            output[:,:,:,nb] = m_transp
            # end
        end
        t_1 += t1_curr
        t_2 += t2_curr
        t_3 += t3_curr
        a_1 += a1_curr
        a_2 += a2_curr
        a_3 += a3_curr
    end
    println("Init_ALLOC : $(t_a)s, $(a_a/1000000) M");
    println("Init_KERNL : $(t_rshp)s, $(a_rshp/1000000) M");
    println("IM2COL     : $(t_1)s, $(a_1/1000000) M");
    println("MATMUL     : $(t_2)s, $(a_2/1000000) M");
    println("RESHAPE    : $(t_3)s, $(a_3/1000000) M");

    return output
end

function caffe_conv4d(x::tensor4, kern::tensor4, bias::Array{Float64, 1}, inner::Bool;stride=1)
    _, t_a, a_a, _, _ = @timed begin
        b, c,  w,   h   = size(x)
        f, c2, k_w, k_h = size(kern)
        o_w, o_h       = w+k_w-1, h+k_h-1
        @assert c2 == c

        # Initialize memory
        m_img    = zeros(o_w*o_h,   c*k_w*k_h)
        m_ker    = zeros(c*k_w*k_h, f)
        m_conved = zeros(o_w*o_h,   f)

         # update to have (b,w,h,c)
        output   = zeros(o_w, o_h, f, b)

        # Put C at the last place of the codes, so that we have (w, h, c, b)
        x_p = zeros(w, h, c, b)
        permutedims!(x_p, x, [3,4,2,1])
        # For each batch, fill m_img, and make computation
        kernel = (k_w, k_h)
    end

    # Fill m_ker
    _, t_rshp, a_rshp, _, _ = @timed for col = 1:f
        m_ker[:,col] = reshape(kern[col,:,:,:], 1,c*k_w*k_h) # flatten a 3D-kernel
    end


    t_1, t_2, t_3 = 0., 0., 0.
    a_1, a_2, a_3 = 0., 0., 0.
    for nb = 1:b
        _, t1_curr, a1_curr, _, _ = @timed im2col_impl(x_p[:,:,:,nb], m_img, kernel, (0,0), (stride,stride))
        _, t2_curr, a2_curr, _, _ = @timed A_mul_B!(m_conved, m_img, m_ker)
        # col2im_impl(m_conved, m_transp, size(x_p)[1:3], kernel, (0,0), (stride,stride))
        _, t3_curr, a3_curr, _, _ = @timed begin
            m_transp = reshape(m_conved, o_w, o_h, f)
            output[:,:,:,nb] = m_transp
        end
        t_1 += t1_curr
        t_2 += t2_curr
        t_3 += t3_curr
        a_1 += a1_curr
        a_2 += a2_curr
        a_3 += a3_curr
    end
    println("Init_ALLOC : $(t_a)s, $(a_a/1000000) M");
    println("Init_KERNL : $(t_rshp)s, $(a_rshp/1000000) M");
    println("IM2COL     : $(t_1)s, $(a_1/1000000) M");
    println("MATMUL     : $(t_2)s, $(a_2/1000000) M");
    println("RESHAPE    : $(t_3)s, $(a_3/1000000) M");

    if inner
        return output[k_w:end-k_w+1, k_h:end-k_h+1,:,:]
    else
        return output
    end
end

function forward(l::CaffeConvLayer, x::tensor4; kwargs...)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    caffe_conv4d!(l.y, l.tmps_forward, l.x, l.kern, l.bias, true) # inner convolution
    return l.y
end

function backward(l::CaffeConvLayer, dldy::tensor4; kwargs...)
    l.dldy = dldy
    flipped = permutedims(flip(l.kern), [1,2,4,3]) # (kw, kh, f, c)
    f = size(flipped,3)
    caffe_conv4d!(l.dldx, l.tmps_backward, l.dldy, flipped, zeros(f), false) # outter convolution
    return l.dldx
end

function getGradient(l::CaffeConvLayer)
    img    = permutedims(l.x, [1,2,4,3])            # (w,h,c,b)   -> (w,h,b,c)
    kernel = permutedims(flip(l.dldy), [1,2,4,3])   # (ow,oh,f,b) -> (ow,oh,b,f)
    f = size(kernel,4) # TODO: this could be wrong
    caffe_conv4d!(l.k_grad_tmp, l.tmps_gradient, img, kernel, zeros(f), true)
    permutedims!(l.k_grad, l.k_grad_tmp, [1,2,4,3])
    batch_size, depth = size(l.x,4), size(l.x, 3)
    l.b_grad = sum(sum(sum(l.dldy, 4), 2), 1)[1,1,:,1]

    return (l.k_grad, l.b_grad)
end

function getParam(l::CaffeConvLayer)
    return (l.k_grad, l.b_grad)
end

function setParam!(l::CaffeConvLayer, theta::Tuple{Array{Float64}})
    # convention: ret[:,end,1,1] is the gradient for bias

    new_kern, new_bias = theta

    l.k_velc = new_kern - l.kern
    l.kern   = new_kern

    l.b_velc = new_bias - l.bias
    l.bias   = new_bias
end

function getVelocity(l::CaffeConvLayer)
    return (l.k_velc, l.b_velc)
end

bsize= 500
l = CaffeConvLayer(32,(3,3))
X = rand(27, 27, 3,  bsize)
Y = rand(25, 25, 32, bsize)

println("First time (compiling...)")
init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => (27, 27, 3)))
@time y1 = forward(l,X)
@time y1 = backward(l,Y)
@time y1 = getGradient(l)

println("Second time (after compilation) CaffeConvLayer")
X = rand(27, 27, 3,  bsize)
Y = rand(25, 25, 32, bsize)
@time begin
    forward(l,X)
end
@time begin
    backward(l,Y)
end
@time begin
    getGradient(l)
end


type ConvLayer <: LearnableLayer
    pad      :: Int                     # Padding
    stride   :: Int                     # Stride
    filter   :: Int                     # Number of filters
    k_size   :: Tuple{Int, Int}         # (k_width,     k_height)
    x_size   :: Tuple{Int, Int, Int}    # (channel,     img_width,  img_height)
    x        :: Array{Float64, 4}       # (batch_size,  channel,    width,     height)
    y        :: Array{Float64, 4}       # (batch_size,  #filter,    out_width, out_height)
    dldy     :: Array{Float64, 4}       # (batch_size,  #filter,    out_width, out_height)
    dldx     :: Array{Float64, 4}       # (batch_size,  channel,    width,     height)

    kern     :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)
    bias     :: Array{Float64, 1}       # (#filter)
    grad     :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)
    velc     :: Array{Float64, 4}       # (#filter,     channel,    k_width,   k_height)

    function ConvLayer(;filters = 32, kernel=(3,3), padding = 0, stride = 1, init="Uniform")
        nothing # TODO
    end
end

function init(l::ConvLayer; kwargs...)
    """
    Initialize the Convolutional layers. Preallocate all the memories.
    """
    nothing
end

function conv4!(out_mat::Union{SubArray{Float64,4},Array{Float64,4}},
    x::Union{SubArray{Float64,4},Array{Float64,4}}, k::Array{Float64,4}
    bias::Array{Float64}; stride=1, padding = 0, flip=false)
    """
    Use for-loop to compute Convolution. Assume the data is already padded.
        k = (#filter, c, kw, kh)
        x = (#batch,  c, w,  h)
        out_math = (#batch, #filter, ow, oh)
    """
    b, c,  w,  h     = size(x)
    f, c2, kw, kh    = size(k)
    @assert c == c2
    @assert kw == kh && kw % 2 == 1 # Curretly only support odd number of tilers
    scale!(out_mat, 0)  # Refill the whole answer.
    kc = kw - 1
    padding = max(padding, kc)
    kc-= padding

    for b_idx = 1:b
    for f_idx = 1:f
        for c_idx = 1:c
            if flip
                ret = conv2(x[b_idx, c_idx, :,:], k[f_idx, c_idx, :,:])[1+kc:end-kc,1+kc:end-kc]
            else
                ret = conv2(x[b_idx, c_idx, :,:], k[f_idx, c_idx, :,:])[1+kc:end-kc,1+kc:end-kc]
            end
            out_mat[b_idx, f_idx, :,:] += ret[1:stride:end,1:stride:end]
        end
        out_mat[b_idx, f_idx, :,:] += bias[f_idx]
    end
    end

    return out_mat
end

function flip(x::Union{SubArray{Float64,4},Array{Float64,4}})
    return x[:,:,end:-1:1, end:-1:1]
end

function forward(l::ConvLayer, x::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    l.x = x
    conv4!(l.y, l.x, l.kern, l.bias; stride = l.stride, padding = l.pad)
    return l.y
end

function backward(l::ConvLayer, dldy::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    l.dldy = dldy
    conv4!(l.dldx, l.dldy, flip(l.kern), zeros(l.filter);) # TODO: strides? padding? bias?
    return l.dldx
end

function gradient(l::ConvLayer)
    batch_size = size(l.x)[1]
    conv4!(l.gradient, l.x, flip(l.dldy), zeros(batch_size);) # TODO: strides? paddings? bias?
    return l.gradient
end

function getParam(l::ConvLayer)
end

function setParam!(l::ConvLayer, theta::Array{Float64})
end

function getVelocity(l::ConvLayer)
end

# Contains helper function for convolutions
Arr4 = Union{SubArray{Float64,4},Array{Float64,4}}
Arr2 = Union{SubArray{Float64,2},Array{Float64,2}}

function conv2!(O::Arr2, A::Arr2, K::Arr2)
    # Inplace version of the covolution (shouldn't take as much memory as conv2)
    
end



function xlowering(l::ConvLayer, x::Union{SubArray{Float64,4},Array{Float64,4}})
    """
    [xlowering(y, x)] lowering the input [x] into GEMM computable shape

    [pred]  [x] has size: (b,c,w,h)

    Then it will reshape [x] into [y] according to the padding, stride, and
    kernel setting of [l]

    [post] return [y] which has size: (n*b,m*b) where [n] is number of window,
           and [m] is the kernel size = [img_d]*[k_w]*[k_h]

    Reference:
    1. https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    2. https://www.nervanasys.com/winograd-2/
    """
    nothing  # TODO
end

function klowering(l::ConvLayer, k::Union{SubArray{Float64,4},Array{Float64,3}})
    """
    [klowering(y, x)] lowering the kernel [k] into GEMM computable shape

    [pred]  [k] has size (f, w, h)

    Then it will reshape each filter into a vector.

    [post] return [y] that has size (b*f, b*w*h*d) where [d] is the depth of input

    Reference:
    1. https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    2. https://www.nervanasys.com/winograd-2/
    """
    nothing # TODO
end

function lyfting(l::ConvLayer, z::Union{SubArray{Float64,4},Array{Float64,3}})
    """
    [lyfting(l,z)] will lyft the GEMM output [z] into the correct output shape

    [pred]  [z] has size (b*[win], b*[f]) where [win] is number of window,
            and [f] is number of filters, [b] is number of batches.

    The function will reshape the [z] to construct the output.

    [post]  return [y] with size (b, f, w_out, h_out) where [b] is batch size
            [f] is number of filters, [w_out] is the output width for [l], and
            [h_out] is the output height for [l].
    """
    nothing # TODO
end

function gemm_conv(out_mat::Union{SubArray{Float64,4},Array{Float64,4}},
    l::ConvLayer, x::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    """
    Use BLAS GEMM function to compute the forward covolution.
    """
    nothing # TODO
end

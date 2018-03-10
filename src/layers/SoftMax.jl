type SoftMax <: Nonlinearity
    base :: LayerBase

    x  :: Array{Float64}        # input m by n matrix which represents m instances with scores for n classes
    jacobian :: Array{Float64}  # cache for the jacobain matrices used in backward
    dldx_cache :: Array{Float64} # cache for dldx
    ly :: Array{Float64}        # cache for row matrix during backward
    lexp :: Array{Float64}      # cache for exponential of l.x in forward
    lsum :: Array{Float64}      # cache for calculating exponential sum in forward

    function SoftMax(config::Dict{String,Any})
        layer = new(
            LayerBase(),
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
        @assert isa(config["input_size"], Int)
        init(layer, (config["batch_size"], config["input_size"]))
        layer
    end

    function SoftMax(prev::Union{Layer,Void})
        layer = new(
            LayerBase(),
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
        connect(layer, [prev])
        init(layer, getOutputSize(prev))
        layer
    end
end

function init(l::SoftMax, out_size::Tuple)
    l.x = Array{Float64}(out_size)
    l.base.y = Array{Float64}(out_size)
    m,n = out_size
    l.jacobian = Array{Float64}(n,n)
    if length(l.base.parents)>0
      l.dldx = Array{Float64}(out_size)
    end
    l.dldx_cache = Array{Float64}(out_size)
    l.ly = Array{Float64}(n)
    l.lexp = Array{Float64}(out_size)
end

function update(l::SoftMax, out_size::Tuple)
    l.x = Array{Float64}(out_size)
    l.base.y = Array{Float64}(out_size)
    m,n = out_size
    l.jacobian = Array{Float64}(n,n)
    if length(l.base.parents)>0
        l.dldx = Array{Float64}(out_size)
    end
    l.dldx_cache = Array{Float64}(out_size)
    l.ly = Array{Float64}(n)
    l.lexp = Array{Float64}(out_size)
end

function forward(l::SoftMax, X::Array{Float64,2}; kwargs...)
    @assert size(X, 2) == size(l.x, 2)
    m,n = size(X)
    if m != size(l.x, 1)
      update(l, size(X))
    end
    l.x = X
    # iterating each row/picture
    l.x = l.x .- maximum(l.x, 2)
    l.lexp = exp.(l.x)
    l.lsum = sum(l.lexp, 2)
    broadcast!(/, l.base.y, l.lexp, l.lsum)
    return l.base.y

end

function backward(l::SoftMax, DLDY::Array{Float64, 2}; kwargs...)
    # credits: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network?newreg=d1e89b443dd346ae8bccaf038a944221
    @assert size(l.base.y) == size(DLDY)
    m,n =size(l.x)
    for batch=1:m
        l.ly = l.base.y[batch,:]
        l.jacobian .= -l.ly .* l.ly'
        l.jacobian[diagind(l.jacobian)] .= l.ly.*(1.0.-l.ly)
        # # n x 1 = n x n * n x 1
        l.dldx_cache[batch,:] = l.jacobian * DLDY[batch,:]
    end
    if length(l.base.parents)>0
        l.base.dldx[l.base.parents[1].base.id] = l.dldx_cache
    end
    return l.dldx_cache
end

type SoftMax <: Nonlinearity
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init  :: Bool
    id        :: Base.Random.UUID

    x  :: Array{Float64}        # input m by n matrix which represents m instances with scores for n classes
    y :: Array{Float64}         # output m by n matrix which uses softmax to normalize the input matrix
    jacobian :: Array{Float64}  # cache for the jacobain matrices used in backward
    dldx :: Array{Float64}      # cahce for the backward result
    ly :: Array{Float64}        # cache for row matrix during backward
    lexp :: Array{Float64}      # cache for exponential of l.x in forward
    lsum :: Array{Float64}      # cache for calculating exponential sum in forward

    function SoftMax()
        return new(
            Layer[], Layer[], false, Base.Random.uuid4(),
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],Float64[])
    end

    function SoftMax(prev::Union{Layer,Void}; config::Union{Dict{String,Any},Void}=nothing, kwargs...)
        layer = new(
            Layer[], Layer[], false, Base.Random.uuid4(),
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],Float64[])
        init(layer, prev, config; kwargs...)
        layer
    end
end

function init(l::SoftMax, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
  if !isa(p,Void)
    l.parents = [p]
    push!(p.children, l)
  end

  if p == nothing
      @assert ndims(config["input_size"]) == 1 # TODO: maybe an error message?
      out_size = (config["batch_size"], config["input_size"][1])
  else
      out_size = getOutputSize(p)
  end
  l.x = Array{Float64}(out_size)
  l.y = Array{Float64}(out_size)
  l.has_init = true;
  m,n = out_size
  l.jacobian = Array{Float64}(n,n)
  l.dldx = Array{Float64}(out_size)
  l.ly = Array{Float64}(n)
  l.lexp = Array{Float64}(out_size)
  l.has_init = true
end

function update(l::SoftMax, out_size::Tuple)
    l.x = Array{Float64}(out_size)
    l.y = Array{Float64}(out_size)
    l.has_init = true;
    m,n = out_size
    l.jacobian = Array{Float64}(n,n)
    l.dldx = Array{Float64}(out_size)
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
    # l.y = l.lexp./l.lsum
    broadcast!(/, l.y, l.lexp, l.lsum)
    return l.y

end

function backward(l::SoftMax, DLDY::Array{Float64, 2}; kwargs...)
    # credits: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network?newreg=d1e89b443dd346ae8bccaf038a944221
    @assert size(l.y) == size(DLDY)
    m,n =size(l.x)
    for batch=1:m
      l.ly = l.y[batch,:]
      l.jacobian .= -l.ly .* l.ly'
      l.jacobian[diagind(l.jacobian)] .= l.ly.*(1.0.-l.ly)
      # # n x 1 = n x n * n x 1
      l.dldx[batch,:] = l.jacobian * DLDY[batch,:]
    end
    return l.dldx

end

type SoftMax <: Nonlinearity
    x  :: Array{Float64}
    y :: Array{Float64}
    has_init :: Bool
    jacobian :: Array{Float64}
    dldx :: Array{Float64}
    # last_loss   :: Array{Float64}

    function SoftMax()
        return new(Float64[], Float64[])
    end
end

function init(l::SoftMax, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
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
end

function forward(l::SoftMax, X::Array{Float64,2}; kwargs...)
	  # X = broadcast(+, X, -maximum(X))
    l.x=X
    m,n = size(X)
    Y = zeros(m,n)
    # iterating each row/picture
    for i=1:m
      exp_sum = 0
      # find the exponential sum of output for each class for this row/picture
      for j=1:n
        exp_sum+=exp(X[i,j])
      end
      # softmaxly normalizing the score
      for j=1:n
        Y[i,j]=exp(X[i,j])/exp_sum
      end
    end
    l.y = Y
    return Y
end

function backward(l::SoftMax, DLDY::Array{Float64}; kwargs...)
    # credits: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network?newreg=d1e89b443dd346ae8bccaf038a944221
    m,n =size(l.x)

    ly = Array{Float64}(n)
    @time for batch=1:m
      ly = l.y[batch,:]

      for i=1:n
        li = ly[i]
        l.jacobian[:,i] = -li * ly
        l.jacobian[i,i] = li*(1-li)
      end

      # l.jacobian = ly'.*repmat(ly, 1, n)
      # for i=1:n
      #   li = l.y[batch,i]
      #   l.jacobian[i,i] = li*(1.0-li)
      # end

      # # n x 1 = n x n * n x 1
      l.dldx[batch,:] = l.jacobian * DLDY[batch,:]
    end

    return l.dldx

end

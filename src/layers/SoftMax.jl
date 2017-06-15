type SoftMax <: Nonlinearity
    x  :: Array{Float64}
    y :: Array{Float64}
    has_init :: Bool
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
    # println(typeof(jacobian))
    result = zeros(m,n)

    for batch=1:m
      jacobian = zeros(n,n)

      # diagonal
      for i=1:n
        jacobian[i,i] = l.y[batch,i]*(1.0-l.y[batch,i])
      end

      # the rest; matrix is symmetric
      for i=1:n
        for j=i+1:n
          term = -l.y[batch,i]*l.y[batch,j]
          jacobian[i,j] = term
          jacobian[j,i] = term
        end
      end
      # n x 1 = n x n * n x 1
      result[batch,:] = jacobian * DLDY[batch,:]
    end

    return result
    # sumX = sum(exp(X))
    # u = zeros(ndims(X), ndims(X))
    # z = zeros(ndims(X))
    # for i = 1: ndims(X)
    #     z[i] = (X[i]/sumX)
    # end
    #
    # for i = 1: ndims(X)
    #     t = z[i]
    #     w = zeros(ndims(X))
    #     w[i] = 1
    #     w = w .- z
    #     u[:,i] = t * w
    # end
    # l.last_loss = DLDY' * u

end

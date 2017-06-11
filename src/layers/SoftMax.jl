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
      # [l] is the first layer, batch_size used default network batch_size
      # and input_size should be single dimensional (i.e. vector)
      @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
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
        # if isnan(Y[i,j])
        #   println(X[i,j])
        #   println(exp_sum)
        #   println()
        # end
      end
    end
    l.y = Y
    return Y
end

function backward(l::SoftMax, DLDY::Array{Float64}; kwargs...)
    # credits: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network?newreg=d1e89b443dd346ae8bccaf038a944221
    X = l.x
    jacobian = zeros(size(l.x)[1], size(l.x)[1])
    # println(typeof(jacobian))
    for i=1:size(l.x)[1]
      for j=1:size(l.x)[1]
        if i==j
          jacobian[i,j] = l.x[j]*(1.0-l.x[j])
        else
          jacobian[i,j] = -l.x[i]*l.x[j]
        end
      end
    end
    # println(DLDY)
    return jacobian * DLDY
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

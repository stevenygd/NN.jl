type CrossEntropyLoss <: LossCriteria
  x :: Array{Float64}
  loss :: Array{Float64}
  classes :: Int
  pred :: Array{Float64}
  dldx :: Array{Float64}

  function CrossEntropyLoss()
    return new(Float64[], Float64[], 0)
  end
end

function init(l::CrossEntropyLoss, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
  if p == nothing
      @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
      out_size = (config["batch_size"], config["input_size"][1])
  else
      out_size = getOutputSize(p)
  end
  l.classes = out_size[2]
  l.x      = Array{Float64}(out_size)
  l.loss   = Array{Float64}(out_size[1])
  l.pred   = Array{Float64}(out_size[1])
  l.dldx   = Array{Float64}(out_size)
end

function convert_to_one_hot(l::CrossEntropyLoss, old_label::Array{Int, 2})
  m,n = size(old_label)
  new_label=zeros(Int64, m, l.classes)
  for i=1:m
    new_label[i, old_label[i]+1] = 1
  end
  return new_label
end

"""
Invaraint for this to work:
  Label needs to be either a vector specifying each item's class or a matrix made up with multiple one hot vectors.
  In the former case, labels need to be in the range of 0:classes-1.
"""
function forward(l::CrossEntropyLoss, Y::Array{Float64,2}, label::Array{Int, 2}; kwargs...)
  if size(label)[2] == 1
    # convert one-dim label to one hot vectors
    label = convert_to_one_hot(l,label)
  end
  # from now on label is guaranteed to be of one-hot
  @assert size(Y) == size(label)
  m,n = size(Y)
  l.loss = zeros(m)
  for i=1:m
    log_sum = 0
    for j=1:n
      p = Y[i,j]
      q = label[i,j]
      if q!=0
        log_sum+=q*log(q/p)
      end
    end
    l.loss[i]=log_sum
  end
  l.x = Y
  # generate prediction
  for i=1:m
    l.pred[i] = findmax(Y[i,:])[2]-1
  end
  return l.loss, l.pred
end

function backward(l::CrossEntropyLoss, label::Array{Int, 2};kwargs...)
  if size(label)[2] == 1
    # convert one-dim label to one hot vectors
    label = convert_to_one_hot(l,label)
  end

  @assert size(l.x) == size(label)

  l.dldx = -label./l.x
  return l.dldx
end

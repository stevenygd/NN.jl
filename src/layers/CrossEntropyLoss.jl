type CrossEntropyLoss <: LossCriteria
  x :: Array{Float64}
  y :: Array{Float64}
  classes :: Int64

  function CrossEntropyLoss()
    return new(Float64[], Float64[], 0)
  end
end

function init(l::CrossEntropyLoss, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
  if p==nothing || typeof(p)<:InputLayer
    error("Loss functions cannot be the first layer; makes no sense")
  end
  out_size = getOutputSize(p)
  l.classes = out_size[2]
  l.x      = Array{Float64}(out_size)
  l.y      = Array{Float64}(out_size)
end

function convert_to_one_hot(label::Array{Int, 2})
  m = size(label)[1]
  new_label::Array{Int,2}
  new_label=zeros(Int64,size(label,2)[1],l.classes)
  for i=1:m
    new_label[i][label[i]+1] = 1;
  end
end

"""
Invaraint for this to work:
  Label needs to be either a vector specifying each item's class or a matrix made up with multiple one hot vectors.
  In the former case, labels need to be in the range of 0:classes-1.
"""
function forward(l::CrossEntropyLoss, Y::Array{Float64,2}, label::Array{Int, 2}; kwargs...)
  if size(label)[2] == 1
    # convert one-dim label to one hot vectors
    label = convert_to_one_hot(label)
  end
  # from now on label is guaranteed to be of one-hot
  @assert size(Y) = size(label)
  loss = zeros(m)
  m,n = size(Y)
  for i=1:m
    log_sum = 0;
    for j=1:n
      p = Y[i,j]
      q = label[i,j]
      log_sum+=q*log(q/p)
    end
    loss = log_sum/n
    loss[i]=loss
  end
  x = Y;
  y = loss;
  # generate prediction
  pred = zeros(m)
  for i=1:m
    pred[i] = findmax(Y[i,:])[2]-1
  end
  return loss, pred
end

"""
for each row x, let x_i be j^th element, loss(x)=log(q_i/x_i)/n+...(other elements)
thus d(loss_j)/dx_j=1/n*x_j/q = x_j/(q_j*n)
where j is the num of classes,
"""
function backward(l::CrossEntropyLoss, label::Array{Int, 2};kwargs...)
  dldx = zeros(classes)
  m,n=size(l.x)
  for i=1:m
    for j=:1:n
      dlidx=l.x[i,j]/(label[i,j]*classes)
      dldlx[j]+=dlidx
    end
  end
  return dldx
end

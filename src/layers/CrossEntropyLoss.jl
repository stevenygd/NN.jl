type CrossEntropyLoss <: LossCriteria
  x :: Array{Float64}       # input normalized matrix, expected to be free of zeros
  loss :: Array{Float64}    # output of current layer; calculate loss of each instance
  classes :: Int            # number of classes in the model
  pred :: Array{Float64}    # cache for prediction result in forward
  dldx :: Array{Float64}    # cache for derivative matrix in backward

  function CrossEntropyLoss()
    return new(Float64[], Float64[], 0, Float64[], Float64[])
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

"""
Requried: label needs to be a matrix that assigns a score for each class for each instance
"""
function forward(l::CrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; smoof = 0.000000000001, kwargs...)

  @assert size(Y) == size(label)
  @assert size(l.x) == size(Y)

  l.x = log(Y)
  l.x = -label.*l.x
  l.loss = sum(l.x,2)
  println()
  l.x = Y

  # generate prediction
  for i=1:size(Y,1)
    l.pred[i] = findmax(Y[i,:])[2]-1
  end

  return l.loss, l.pred
end

function backward(l::CrossEntropyLoss, label::Array{Float64, 2};kwargs...)
  @assert size(l.x) == size(label)

  l.dldx = -label./l.x
  return l.dldx
end

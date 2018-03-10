type CrossEntropyLoss <: LossCriteria

  base     :: LayerBase

  x :: Array{Float64}       # input normalized matrix, expected to be free of zeros
  loss :: Array{Float64}    # output of current layer; calculate loss of each instance
  dldx :: Array{Float64}    # cache for derivative matrix in backward

  function CrossEntropyLoss(config::Union{Dict{String,Any},Void}, label::DataLayer)
    layer = new(
      LayerBase(),
      Float64[], Float64[], Float64[])
    @assert isa(config["input_size"], Int)
    init(layer,(config["batch_size"], config["input_size"]))
    layer
  end

  function CrossEntropyLoss(prev::Union{Layer,Void}, label::DataLayer)
    layer = new(
      LayerBase(),
      Float64[], Float64[], 0,  Float64[])
    connect(layer, [prev])
    init(layer,getOutputSize(prev))
    layer
  end
end

function init(l::CrossEntropyLoss, out_size::Tuple)
  l.x      = Array{Float64}(out_size)
  l.loss   = Array{Float64}(out_size[1])
  l.dldx   = Array{Float64}(out_size)
end

function update(l::CrossEntropyLoss, out_size::Tuple)
  l.x = Array{Float64}(out_size)
  l.loss   = Array{Float64}(out_size[1])
  l.dldx = Array{Float64}(out_size)
end

"""
Requried: label needs to be a matrix that assigns a score for each class for each instance
"""
function forward(l::CrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)

  @assert size(Y, 2) == size(l.x, 2)
  m,n = size(Y)
  if m != size(l.x, 1)
    update(l, size(Y))
  end

  l.x = log.(Y)
  l.x = -label.*l.x
  # broadcast!(*, -l.x, label)
  l.loss = sum(l.x,2)
  l.x = Y

  return l.loss, l.x
end

function backward(l::CrossEntropyLoss, label::Array{Float64, 2};kwargs...)
  @assert size(l.x) == size(label)
  l.dldx = -label./l.x
  return l.dldx
end

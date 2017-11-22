include("InputLayer.jl")

type SoftMaxCrossEntropyLoss <: LossCriteria
    base    :: LayerBase

    exp    :: Array{Float64} # cache for exp(x)
    lsum   :: Array{Float64} # cache for sum(exp,2)
    loss   :: Array{Float64} # output of cross entropy loss
    pred   :: Array{Int64}   # output for prediction

    function SoftMaxCrossEntropyLoss()
        return new(LayerBase(), Float64[], Float64[], Float64[], Int64[])
    end

    function SoftMaxCrossEntropyLoss(prev::Union{Layer,Void}; config::Union{Dict{String,Any},Void}=nothing, kwargs...)
        layer = new(LayerBase(), Float64[], Float64[], Float64[], Int64[])
        init(layer,prev, config;kwargs...)
        layer
    end
end

function init(l::SoftMaxCrossEntropyLoss, p::Union{Layer,Void}, config::Union{Dict{String,Any},Void}; kwargs...)

    # TODO: currently I only accept Single dimensional dropout
    if p == nothing
        # [l] is the first layer, batch_size used default network batch_size
        # and input_size should be single dimensional (i.e. vector)
        @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
    else
        out_size = getOutputSize(p)
    end
    N, D     = out_size

    # loss layer's parents[1] would be an input layer providing label
    label = InputLayer((out_size); tag="labels")
	if !isa(p,Void)
        parents = [label, p]
    else
        parents = [label]
    end
    connect(l, parents)

    foreach(x -> l.base.dldx[x.base.id] = Array{Float64}(out_size), l.base.parents)
    l.base.x      = Array{Float64}(out_size)
    l.base.y      = Array{Float64}(out_size)
    l.loss   = Array{Float64}(N)
    l.exp    = Array{Float64}(out_size)
    l.pred   = Array{Int64}(N)
    l.lsum   = Array{Float64}(N)
end

function update(l::SoftMaxCrossEntropyLoss, input_size::Tuple;)
    # We only allow to update the batch size
    @assert length(input_size) == 2
    @assert input_size[2] == size(l.base.x, 2)
    N, D = input_size[1], size(l.base.x, 2)
    foreach(x -> l.base.dldx[x.base.id] = Array{Float64}(input_size), l.base.parents)
    l.base.x      = Array{Float64}(input_size)
    l.base.y      = Array{Float64}(input_size)
    l.loss   = Array{Float64}(N)
    l.pred   = Array{Int64}(N)
    l.lsum = Array{Float64}(N)
    # update(l.base.parents[1], input_size)
end

function forward(l::SoftMaxCrossEntropyLoss; kwargs...)
    forward(l,l.base.parents[2].base.y, l.base.parents[1].base.y; kwargs...)
end

function forward(l::SoftMaxCrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)
    @assert size(Y, 2) == size(l.base.x, 2)
    m,n = size(Y)
    if m != size(l.base.x, 1)
      update(l, size(Y))
    end
    l.base.x = Y
    l.exp = exp.(l.base.x)
    l.lsum = sum(l.exp,2)
    l.base.y = l.exp ./ l.lsum

    for i=1:m
      for j=1:n
        l.exp[i,j] = exp(l.base.x[i,j])
      end
      l.lsum[i] = sum(l.exp[i,:])
      for j=1:n
        l.base.y[i,j] = l.exp[i,j]/l.lsum[i]
        l.exp[i,j] = log(l.base.y[i,j])*label[i,j]
      end
      l.loss[i] = -sum(l.exp[i,:])
    end


    # l.loss = - sum(log(l.y) .* label,2)

    return l.loss, l.base.y
end

function backward(l::SoftMaxCrossEntropyLoss;kwargs...)

    label = l.base.parents[1].base.y
    for p ∈ l.base.parents
        parent_id = p.base.id
        l.base.dldx[parent_id] = l.base.y .* sum(label,2) - label
    end
end

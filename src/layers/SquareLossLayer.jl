type SquareLossLayer <: LossCriteria
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init  :: Bool
    id        :: Base.Random.UUID

    x :: Array{Float64}
    y :: Array{Float64}
    function SquareLossLayer()
        return new(Layer[], Layer[], false, Base.Random.uuid4(),
            Float64[], Float64[])
    end
    function SquareLossLayer(prev::Union{Layer,Void}; config::Union{Dict{String,Any},Void}=nothing, kwargs...)
        layer = new(Layer[], Layer[], false, Base.Random.uuid4(),
            Float64[], Float64[])
        init(layer, prev, config;kwargs...)
        layer
    end
end

function init(l::SquareLossLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    if !isa(p,Void)
        l.parents = [p]
        push!(p.children, l)
    end

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
    l.x      = Array{Float64}(out_size)
    l.y      = Array{Float64}(out_size)

    l.has_init = true
end

function forward(l::SquareLossLayer, Y::Array{Float64}, t::Array{Float64}; kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    @return:
    loss:
    //TODO:
    pred: naive prediction that if greater than 0 then 1 else 0

    """
    N = convert(Float64, size(Y)[1])
    l.x = Y

    local temp = Y - t
    local loss = temp' * temp / 2.
    local pred = map(x -> (x > 0)?1:-1, Y)
    return loss, pred

end

function backward(l::SquareLossLayer, t::Array{Float64}; kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local N = convert(Float64, size(l.x)[1])
    local DLDY = l.x - t
    return DLDY
end

#l = SquareLossLayer()
#Y = zeros(4,1)
#Y[1,:] = 0
#Y[2,:] = 1
#Y[3,:] = 3
#Y[4,:] = 5
#t = zeros(4,1)
#t[1,:] = 0
#t[2,:] = 1
#t[3,:] = 1
#t[4,:] = 0

#loss, pred = forward(l, Y, t)
#println((loss, pred))
#println(backward(l, t))

# Define the ReLu layers
include("LayerBase.jl")

type ReLu <: Nonlinearity
    parents  :: Array{Layer}
    children :: Array{Layer}
    has_init :: Bool
    id       :: Base.Random.UUID

    alpha    :: Float64
    x        :: Array{Float64}
    y        :: Array{Float64}
    dldx     :: Dict{Base.Random.UUID, Array{Float64}}
    dldy     :: Array{Float64}

    function ReLu(alpha::Float64 = 1.0)
        @assert alpha >= 0.
        return new(Layer[], Layer[], false, Base.Random.uuid4(), alpha, Float64[], Float64[], Float64[], Float64[])
    end

    function ReLu(prev::Union{Layer,Void}; config::Union{Dict{String,Any},Void}=nothing, alpha::Float64 = 1.0, kwargs...)
        @assert alpha >= 0.
        layer = new(Layer[], Layer[], false, Base.Random.uuid4(), alpha, Float64[], Float64[], Float64[], Float64[])
        init(layer, prev, config; kwargs...)
        layer
    end
end

function init(l::ReLu, p::Union{Layer,Void}, config::Union{Dict{String,Any},Void}; kwargs...)
    # TODO: currently I only accept Single dimensional dropout
    if !isa(p,Void)
        l.parents = [p]
        push!(p.children, l)
    end

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
    l.dldx = Array{Float64}(out_size)
    l.dldy = Array{Float64}(out_size)

    l.has_init = true
end

function update(l::ReLu, input_size::Tuple;)
    l.x = Array{Float64}(input_size)
    l.y = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.dldy = Array{Float64}(input_size)
end

function forward(l::ReLu; kwargs...)
    forward(l, l.parents[1].y; kwargs...)
end

function forward(l::ReLu, X::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    if size(l.x) != size(X)
        update(l, size(X))
    end
    l.x = X
    broadcast!(max, l.y, X, 0.)
    broadcast!(*,   l.y, l.y, l.alpha)
    return l.y
end

function backward(l::ReLu; kwargs...)
    DLDY = sum(map(x -> x.dldx[l.id], l.children))
    @assert size(l.x) == size(DLDY)
    # if size(l.dldx, 1) != size(DLDY, 1)
    #     l.dldx = Array{Float64}(size(DLDY))
    # end
    l.dldy = DLDY
    parent_id = l.parents[1]
    broadcast!(>, l.dldx[parent_id], l.x, 0.)        # l.dldx = l.x .> 0.
    broadcast!(*, l.dldx[parent_id], l.dldx[parent_id], l.alpha)    # l.dldx = l.dldx * alpha
    broadcast!(*, l.dldx[parent_id], l.dldx[parent_id], DLDY)
    return l.dldx
end


# l = ReLu()
# X = rand(1000, 500)
# Y = rand(1000, 500)
# println("Compile the method for the first time...")
# @time init(l, nothing, Dict{String, Any}("batch_size" => 1000, "input_size" => [500]))
# @time forward(l,X)
# @time backward(l,Y)
#
# println("Start profiling...")
# print("Forward:")
# @time begin
#   for i = 1:10
#     forward(l,X)
#   end
# end
#
# @time begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end
#
# print("Backward")
# @time begin
#   for i = 1:10
#     backward(l,Y)
#   end
# end
#
# @time begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end

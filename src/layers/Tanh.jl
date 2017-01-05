# include("LayerBase.jl")
type Tanh <: Nonlinearity
    has_init    :: Bool
    x           :: Array{Float64}
    y           :: Array{Float64}
    dldx        :: Array{Float64}
    dldy        :: Array{Float64}

    function Tanh()
        return new(false, Float64[], Float64[], Float64[], Float64[])
    end
end

function init(l::Tanh, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    # TODO: currently I only accept Single dimensional dropout
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

function forward(l::Tanh, X::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    l.x  = X
    l.y = tanh(X)
    return l.y
end

function backward(l::Tanh, DLDY::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    @assert size(l.x) == size(DLDY)
    l.dldx = (1 - l.y .* l.y) .* DLDY #d(tanh(x))/dx = 1 - tanh(x)^2
    return l.dldx
end

# l = Tanh()
# X = [ 1. 2; -1 3; 1 -2; -3 -3]
# Y = [ 2. 3; 2 5; 3 6; 2 2]
# println(forward(l, X))
# println(backward(l, Y))

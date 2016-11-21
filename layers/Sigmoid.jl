include("Base.jl")

type Sigmoid <: Nonlinearity
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}

    function Sigmoid()
        return new(Float64[], Float64[], Float64[])
    end
end

function forward(l::Sigmoid, X::Array{Float64}; kwargs...)
    l.last_input  = X
    l.last_output = map(x -> 1/(1+ e^(-x)), X)
    return l.last_output
end

function backward(l::Sigmoid, DLDY::Array{Float64}; kwargs...)
    @assert size(l.last_input) == size(DLDY)
    l.last_loss = l.last_output .* (1 - l.last_output) .* DLDY # d(sigmoid(x))/dx = sigmoid(x)(1 - sigmoid(x))
    return l.last_loss
end

function gradient(l::Sigmoid)
    0
end

function getParam(l::Sigmoid)
    0
end

function setParam!(l::Sigmoid, theta)
    nothing
end

function getVelocity(l::Sigmoid)
    0
end

include("Base.jl")

type Tanh <: Nonlinearity
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}

    function Tanh()
        return new(Float64[], Float64[], Float64[])
    end
end

function forward(l::Tanh, X::Array{Float64}; kwargs...)
    l.last_input  = X
    l.last_output = tanh(X)
    return l.last_output
end

function backward(l::Tanh, DLDY::Array{Float64}; kwargs...)
    @assert size(l.last_input) == size(DLDY)
    l.last_loss = (1 - l.last_output .* l.last_output) .* DLDY #d(tanh(x))/dx = 1 - tanh(x)^2
    return l.last_loss
end

# l = Tanh()
# X = [ 1. 2; -1 3; 1 -2; -3 -3]
# Y = [ 2. 3; 2 5; 3 6; 2 2]
# println(forward(l, X))
# println(backward(l, Y))

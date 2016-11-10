include("Criteria.jl")

# Define the ReLu layers
type ReLu <: Nonlinearity
    alpha       :: Float64
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}
    last_diff   :: Array{Float64}

    function ReLu(alpha::Float64 = 1.0)
        @assert alpha >= 0.
        return new(alpha, Float64[], Float64[], Float64[], Float64[]) 
    end
end

function forward(l::ReLu, X::Array{Float64}; kwargs...)
    l.last_input  = X
    l.last_output = max(X, 0.) * l.alpha
    l.last_output
end

function backward(l::ReLu, DLDY::Array{Float64}; kwargs...)
    @assert size(l.last_input) == size(DLDY)
    l.last_loss = DLDY .* map(x -> x > 0 ? l.alpha : 0.0, l.last_output)
    return l.last_loss
end

function gradient(l::ReLu)
    0
end

function getParam(l::ReLu)
    0
end

function setParam!(l::ReLu, theta)
    nothing
end

function getLDiff(l::ReLu)
    0
end

l = ReLu()
X = [ 1. 2; -1 3; 1 -2; -3 -3]
Y = [ 2. 3; 2 5; 3 6; 2 2]
println(forward(l, X))
println(backward(l, Y))

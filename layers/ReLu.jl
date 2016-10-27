include("Criteria.jl")

# Define the ReLu layers
type ReLu <: Nonlinearity
    alpha       :: Float64
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}
    function ReLu(alpha::Float64 = 1.0)
        @assert alpha >= 0.
        return new(alpha, Float64[], Float64[], Float64[])
    end
end

function forward(l::ReLu, x::Array{Float64})
    l.last_input  = x
    l.last_output = map(y -> max(0., y*l.alpha), x)
    l.last_output
end

function backward(l::ReLu, loss::Array{Float64})
    @assert size(l.last_input) == size(loss)
    l.last_loss = loss
    local input_fil = map(x -> x > 0 ? 1.0 : 0.0, l.last_input)
    return input_fil .* loss
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

l = ReLu()
#println(forward(l, [1.,0.,-1.,2.]))
#println(backward(l, [3.0,2.0,1.,1.0]))

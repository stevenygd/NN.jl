include("Criteria.jl")

# Define the Fully Connected layers
type FCLayer <: Layer
    W           :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}

    function FCLayer(i, o)
        return new(rand(o,i), zeros(i), zeros(o), zeros(o))
    end
end

function forward(l::FCLayer, x::Array{Float64,1})
    @assert ndims(x) == 1 && size(x) == (size(l.W)[2],)
    l.last_input  = x
    l.last_output = l.W * x # matrix multiplication
    return l.last_output
end

function backward(l::FCLayer, loss::Array{Float64,1})
    @assert size(loss) == (size(l.W)[1],)
    l.last_loss = loss
    l.W'*loss
end

function gradient(l::FCLayer)
    @assert size(l.last_loss) == (size(l.W)[1],)
    println("FC Loss:$(l.last_loss),\nFC Last Input:$(l.last_input)")
    l.last_loss * l.last_input'
end

function getParam(l::FCLayer)
    l.W
end

function setParam!(l::FCLayer, theta::Array{Float64})
    @assert size(l.W) == size(theta)
    local diff = abs(l.W - theta)
    if sum(diff) > 0.
        println("Updating W :$(sum(l.W)), $(sum(theta))")
    end
    l.W = theta
end

l = FCLayer(10,20)
forward(l, rand(10))

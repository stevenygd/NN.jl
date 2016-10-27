include("Criteria.jl")

# Define the Fully Connected layers
type FCLayer <: Layer
    W           :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}
    last_diff   :: Array{Float64}

    function FCLayer(i, o)
        # Use Glorot initialization: http://lasagne.readthedocs.io/en/latest/modules/init.html#r5
        local a = sqrt(2. / (i + o))
        local newW = rand(o,i)* 2 * a - a
        return new(newW, zeros(i), zeros(o), zeros(o), zeros(o, i))
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
    l.last_loss * l.last_input'
end

function getParam(l::FCLayer)
    l.W
end

function setParam!(l::FCLayer, theta::Array{Float64})
    @assert size(l.W) == size(theta)
    l.last_diff = theta - l.W
    l.W = theta
end

function getLDiff(l::FCLayer)
    return l.last_diff
end

l = FCLayer(10,20)
forward(l, rand(10))

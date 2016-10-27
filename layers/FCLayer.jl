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
        local newW = rand(i,o)* 2 * a - a
        return new(newW, zeros(i), zeros(o), zeros(o), zeros(o, i))
    end
end

function forward(l::FCLayer, X::Array{Float64,2})
    # X      : NxI matrix, N is the mini batch size, I is the input size
    # Output : NxO matrix
    @assert size(X)[2] == size(l.W)[1]
    l.last_input  = X
    l.last_output = X * l.W # generate NxO matrix O is the output size
    return l.last_output
end

function backward(l::FCLayer, DLDY::Array{Float64,2})
    @assert size(DLDY)[2] == size(l.W)[2]
    l.last_loss = DLDY
    return DLDY * l.W'
end

function gradient(l::FCLayer)
    local g = l.last_input' * l.last_loss
    @assert size(g) == size(l.W)
    return g
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

l = FCLayer(12,20)
println(forward(l, rand(10,12)))
println(backward(l, rand(10,20)))
println(gradient(l))

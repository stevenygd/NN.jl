include("Criteria.jl")

# Define the Fully Connected layers
type FCLayer <: Layer
    i           :: Int64
    W           :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}
    last_diff   :: Array{Float64}

    function FCLayer(i, o; init_type = "Uniform")
        # Use Glorot initialization: http://lasagne.readthedocs.io/en/latest/modules/init.html#r5
        local newW = zeros(i+1,o)
        if init_type == "Uniform"
            local a    = sqrt(12. / (i + o))
            newW = randn(i+1,o)* 2 * a - a
        elseif init_type == "Normal"
            local sigma = sqrt(2. / (i + o))
            newW  = randn(i+1,o)* 2 * sigma
        elseif init_type == "Random"
            newW = rand(i+1,o) - 0.5
        end
        # save the original input size
        return new(i, newW, zeros(i), zeros(o), zeros(o), zeros(i, o))
    end
end

verbose = 0

function forward(l::FCLayer, X::Array{Float64,2})
    # X      : NxI matrix, N is the mini batch size, I is the input size
    # Output : NxO matrix
    @assert size(X)[2] == l.i
    # Pad one at the end of the vector
    l.last_input  = ones(size(X)[1], l.i + 1)
    l.last_input[:,1:l.i]  = X

    l.last_output = l.last_input * l.W
    return l.last_output
end

function backward(l::FCLayer, DLDY::Array{Float64,2})
    @assert size(DLDY)[2] == size(l.W)[2]
    l.last_loss = DLDY
    return (DLDY * l.W')[:, 1:l.i] # get rid of the bias
end

function gradient(l::FCLayer)
    local ret = zeros()
    local g = l.last_input' * l.last_loss
    @assert size(g) == size(l.W)
    return g
end

function getParam(l::FCLayer)
    return l.W
end

function setParam!(l::FCLayer, theta::Array{Float64})
    @assert size(l.W) == size(theta)
    l.last_diff = theta - l.W
    if verbose > 0
        println("Difference:$(sum(abs(l.last_diff)))")
    end
    l.W = theta
end

function getLDiff(l::FCLayer)
    return l.last_diff
end

l = FCLayer(2,3)
println(l)
l.W = [ 2 3 4 ; 0 9 8 ; 1 1 1]
X = [ 1. 2; 3 4 ]
Y = [ 5. 6 7 ; 7 8 9 ]
println(forward(l, X))
println(backward(l,Y))
println(gradient(l))


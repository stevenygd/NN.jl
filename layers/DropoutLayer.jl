include("Criteria.jl")

type DropoutLayer <: Layer
    p :: Float64
    last_drop  :: Array{Float64}
    last_input :: Array{Float64}
    last_output:: Array{Float64}
    last_loss  :: Array{Float64}

    function DropoutLayer(p::Float64)
        return new(p, Float64[], Float64[], Float64[], Float64[])
    end
end

function forward(l::DropoutLayer, x::Array{Float64,2})
    l.last_input  = x
    N, D = size(x)
    l.last_drop   = repeat(map(e -> (e > l.p) ? 1.0 : 0.0, rand(D))', outer=(N,1))
    l.last_output = l.last_input .* l.last_drop
    return l.last_output
end

function backward(l::DropoutLayer, DLDY::Array{Float64})
    @assert size(DLDY) == size(l.last_drop) 
    l.last_loss = DLDY
    return l.last_loss .* l.last_drop
end

function gradient(l::DropoutLayer)
    return 0
end

function getParam(l::DropoutLayer)
    return 0
end

function setParam!(l::DropoutLayer, theta)
    nothing
end

function getLDiff(l::DropoutLayer)
    0
end

l = DropoutLayer(0.3)
X = rand(10, 5)
Y = rand(10, 5)
# println(forward(l, X))
# println(backward(l, Y))
# println(gradient(l))


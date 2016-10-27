include("Criteria.jl")

type DropoutLayer <: Layer
    p :: Float64
    d :: Int64
    last_drop  :: Array{Float64, 2}
    last_input :: Array{Float64, 2}
    last_output:: Array{Float64, 2}
    last_loss  :: Array{Float64, 2}

    function DropoutLayer(p::Float64, d::Int64)
        return new(p, d, zeros(d), zeros(d), zeros(d), zeros(d))
    end
end

function forward(l::DropoutLayer, x::Array{Float64,1})
    @assert size(x)[1] == l.d 
    l.last_input  = x
    l.last_drop   = Diagonal(map(e -> (e > l.p) ? 1.0 : 0.0, rand(l.d)))
    l.last_output = l.last_drop .* l.last_input
    return l.last_output
end

function backward(l::DropoutLayer, DLDY::Array{Float64,1})
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

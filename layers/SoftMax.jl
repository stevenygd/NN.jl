include("Criteria.jl")

type SoftMax <: Nonlinearity
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    last_loss   :: Array{Float64}

    function SoftMax()
        return new(Float64[], Float64[], Float64[]) 
    end
end

function forward(l::SoftMax, X::Array{Float64}; kwargs...)
	X = broadcast(+, X, -maximum(X))
    l.last_input = X
    sumX = sum(exp(X))
    Y = (x ./ sumX)
    l.last_output = Y
    return Y

end

function backward(l::SoftMax, DLDY::Array{Float64}; kwargs...)
    @assert size(l.last_input) == size(DLDY)
    X = l.last_input
    sumX = sum(exp(X))
    u = zeros(ndims(X), ndims(X))
    z = zeros(ndims(X))
    for i = 1: ndims(X)
        z[i] = (X[i]/sumX)
    end

    for i = 1: ndims(X)
        t = z[i]
        w = zeros(ndims(X))
        w[i] = 1
        w = w .- z
        u[:,i] = t * w
    end
    l.last_loss = DLDY' * u
    return l.last_loss
   
end

function gradient(l::SoftMax)
    0
end

function getParam(l::SoftMax)
    0
end

function setParam!(l::SoftMax, theta)
    nothing
end

function getLDiff(l::SoftMax)
    0
end
include("Base.jl")

type DropoutLayer <: Layer
    p :: Float64
    last_drop  :: Array{Float64}
    last_input :: Array{Float64}
    last_output:: Array{Float64}
    last_loss  :: Array{Float64}

    function DropoutLayer(p::Float64)
        @assert abs(p - 1.) >  1e-4 # Basically [p] couldn't be 1
        return new(p, Float64[], Float64[], Float64[], Float64[])
    end
end

function forward(l::DropoutLayer, x::Array{Float64,2}; deterministics=false)
    # Need to rescale the inputs so that the expected output mean will be the same
    l.last_input  = x
    N, D = size(x)
    if deterministics
        l.last_drop = ones(N,D)
    else
        # l.last_drop   = repeat(map(e -> (e > l.p) ? 1.0 : 0.0, rand(D))', outer=(N,1))
        local scale = 1.0 / (1. - l.p)
        l.last_drop = map(e -> (e > l.p) ? scale : 0.0, rand(N,D))
    end
    l.last_output = l.last_drop .* l.last_input
    return l.last_output
end

function backward(l::DropoutLayer, DLDY::Array{Float64}; kwargs...)
    @assert size(DLDY)[2] == size(l.last_drop)[2] &&
            size(DLDY)[1] == size(l.last_input)[1]
    l.last_loss = DLDY
    return broadcast(.*, l.last_loss, l.last_drop)
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

#l = DropoutLayer(0.3)
#X = rand(10, 5)
#Y = rand(10, 5)
# println(forward(l, X))
# println(backward(l, Y))
# println(gradient(l))

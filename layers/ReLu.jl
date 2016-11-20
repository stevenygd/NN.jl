include("Base.jl")

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

function forward(l::ReLu, X::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    if size(l.last_input, 1)  != size(X, 1) ||
       size(l.last_output, 1) != size(X, 1)
       l.last_input = Array{Float64}(size(X))
       l.last_output = Array{Float64}(size(X))
    end
    l.last_input = X
    broadcast!(max, l.last_output, X, 0.)
    broadcast!(*,   l.last_output, l.last_output, l.alpha)
    return l.last_output
end

function backward(l::ReLu, DLDY::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    @assert size(l.last_input) == size(DLDY)
    if size(l.last_loss, 1) != size(DLDY, 1)
        l.last_loss = Array{Float64}(size(DLDY))
    end
    broadcast!(>, l.last_loss, l.last_input, 0.)        # l.last_loss = l.last_input .> 0.
    broadcast!(*, l.last_loss, l.last_loss, l.alpha)    # l.last_loss = l.last_loss * alpha
    broadcast!(*, l.last_loss, l.last_loss, DLDY)
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

#
# l = ReLu()
# X = rand(1000, 500)
# Y = rand(1000, 500)
# println("Compile the method for the first time...")
# forward(l,X)
# backward(l,Y)
#
# println("Start profiling...")
# print("Forward:")
# @time begin
#   for i = 1:1000
#     forward(l,X)
#   end
# end
#
# print("Backward")
# @time begin
#   for i = 1:1000
#     backward(l,Y)
#   end
# end

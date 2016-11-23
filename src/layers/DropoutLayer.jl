type DropoutLayer <: NoiseLayer
    p :: Float64
    last_drop  :: Array{Float64}
    last_input :: Array{Float64}
    last_output:: Array{Float64}
    last_loss  :: Array{Float64}
    last_dldy  :: Array{Float64}

    function DropoutLayer(p::Float64)
        @assert abs(p - 1.) >  1e-4 # Basically [p] couldn't be 1
        return new(p, Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

function forward(l::DropoutLayer, x::Union{SubArray{Float64,2},Array{Float64,2}}; deterministics=false)
    # Need to rescale the inputs so that the expected output mean will be the same
    l.last_input  = x
    N, D = size(x)

    if size(l.last_drop,1)   != N ||
       size(l.last_output,1) != N
        l.last_drop = ones(N,D)
        l.last_output = zeros(N,D)
    end

    rand!(l.last_drop)

    if ! deterministics
        broadcast!(>, l.last_drop, l.last_drop, l.p)
        broadcast!(/, l.last_drop, l.last_drop, 1-l.p)
    end
    broadcast!(*, l.last_output, l.last_drop, l.last_input)
    return l.last_output
end

# Donot annotate DLDY since it could be subarray
function backward(l::DropoutLayer, DLDY::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    @assert size(DLDY)[2] == size(l.last_drop)[2] &&
            size(DLDY)[1] == size(l.last_input)[1]
    if size(l.last_dldy,1) != size(DLDY, 1) ||
       size(l.last_loss,1) != size(DLDY, 1)
        l.last_dldy = Array{Float64}(size(DLDY,1), size(DLDY,2))
        l.last_loss = Array{Float64}(size(DLDY,1), size(DLDY,2))
    end
    l.last_dldy = DLDY
    broadcast!(*, l.last_loss, l.last_dldy, l.last_drop)
    return l.last_loss
end

# l = DropoutLayer(0.3)
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

# Profile.clear()
# Profile.init()
# @profile begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end
# Profile.print()
# println(backward(l, Y))
# println(gradient(l))

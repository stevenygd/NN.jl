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
    # print("DropoutLayer (forward):")
    # @time begin
      l.last_input  = x
      N, D = size(x)

      if size(l.last_drop)[1] != N
        l.last_drop = ones(N,D)
      end

      rand!(l.last_drop)

      if ! deterministics
        l.last_drop = (l.last_drop .> l.p) ./ (1-l.p)
      end

      l.last_output = l.last_drop .* l.last_input
    # end
    return l.last_output
end

function backward(l::DropoutLayer, DLDY::Array{Float64}; kwargs...)
  # print("DropoutLayer (backward):")
  # @time begin
    @assert size(DLDY)[2] == size(l.last_drop)[2] &&
            size(DLDY)[1] == size(l.last_input)[1]
    l.last_loss = DLDY
    local ret = l.last_loss .* l.last_drop
  # end
  return ret
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

# l = DropoutLayer(0.3)
# X = rand(1000, 500)
# Y = rand(1000, 500)
# using IProfile
# forward(l,X)
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

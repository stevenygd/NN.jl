include("Base.jl")

type SoftMaxCrossEntropyLoss <: LossCriteria
    last_loss   :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    function SoftMaxCrossEntropyLoss()
        return new(Float64[], Float64[])
    end
end

function forward(l::SoftMaxCrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    N = size(Y)[1]
    l.last_input = Y
    Y = broadcast(+, Y, - maximum(Y, 2))
    l.last_output = broadcast(+, -Y, log(sum(exp(Y), 2)))

    @assert size(l.last_output) == size(Y)
    label = map(x -> convert(Int64, x) + 1, label)
    local loss = map(i -> l.last_output[i, label[i]], 1:N)
	  local pred = map(i -> findmax(Y[i,:])[2] - 1, 1:N)
    return loss, pred
end

function backward(l::SoftMaxCrossEntropyLoss, label::Array{Float64, 2};kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local N = size(l.last_input)[1]
    local TAR = zeros(size(l.last_input))
    for i = 1:N
        TAR[i, convert(Int64,label[i]) + 1] = 1.
        @assert sum(TAR[i,:]) == 1 && minimum(TAR[i,:]) >= 0
    end

    local Y = l.last_input
    Y = exp(broadcast(+, Y, - maximum(Y,2)))
    Y = broadcast(/, Y, sum(Y,2))
    for i = 1: N
        @assert abs(sum(Y[i,:]) - 1.) <= 1e-6 && minimum(Y[i,:]) >= 0.
    end
    return Y .- TAR
end

# l = SoftMaxCrossEntropyLoss()
# lbl = map(x -> convert(Float64, x), rand(0:9,10))
# println(size(lbl))
# println(lbl)
# println(forward(l, rand(5,10), lbl))
# println(backward(l, lbl))
# y = zeros(2,1)
# y[1] = 1
# y[2] = 2
# x = [ 2. 1. 3;
#     -2. 3. 2.]
#loss, pred = forward(l, x, y)
#println((loss, pred))
#println(backward(l,y))

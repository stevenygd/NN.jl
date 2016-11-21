include("Base.jl")

type SoftMaxCrossEntropyLoss <: LossCriteria
    last_loss   :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    tmpY        :: Array{Float64}
    loss        :: Array{Float64}
    pred        :: Array{Int64}
    label       :: Array{Int64}

    function SoftMaxCrossEntropyLoss()
        return new(Float64[], Float64[], Float64[], Float64[], Float64[],
                   Int64[], Int64[])
    end
end

function forward(l::SoftMaxCrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    N = size(Y, 1)
    if  size(l.last_input, 1) != N || size(l.last_output, 1) != N
        l.last_loss   = Array{Float64}(size(Y))
        l.last_input  = Array{Float64}(size(Y))
        l.last_output = Array{Float64}(size(Y))
        l.tmpY        = Array{Float64}(N, 1)
        l.loss        = Array{Float64}(N)
        l.pred        = Array{Int64}(N)
        l.label       = Array{Int64}(N)
    end

    l.last_input = Y
    maximum!(l.tmpY, Y)
    broadcast!(-, l.last_output, l.last_input, l.tmpY)
    sum!(l.tmpY, exp(l.last_output))
    broadcast!(+, l.last_output, log(l.tmpY), l.last_output)

    @assert size(l.last_output) == size(Y)
    map!(x -> convert(Int64, x) + 1,        l.label, label)
    map!(i -> l.last_output[i, l.label[i]], l.loss,  1:N)
    map!(i -> findmax(Y[i,:])[2] - 1,       l.pred,  1:N)

    return l.loss, l.pred
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

l = SoftMaxCrossEntropyLoss()
X = rand(500, 10) #input size 784, batch size 500
L = map(x -> ceil(x), rand(500, 1))

println("First time (compiling...)")
@time forward(l,X,L)
@time backward(l,L)

println("Second time ...")
@time forward(l,X,L)
@time backward(l,L)


println("Third time (profiling...)")
@time begin
  for i = 1:1000
    forward(l,X,L)
  end
end
@time begin
  for i = 1:1000
    backward(l,L)
  end
end

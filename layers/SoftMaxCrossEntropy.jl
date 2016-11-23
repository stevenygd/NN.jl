include("Base.jl")

type SoftMaxCrossEntropyLoss <: LossCriteria
    last_loss   :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    target      :: Array{Float64}
    loss        :: Array{Float64}
    pred        :: Array{Int64}
    label       :: Array{Int64}
    iter        :: UnitRange{Int64}

    function SoftMaxCrossEntropyLoss()
        return new(Float64[], Float64[], Float64[], Float64[], Float64[], Int64[], 1:1)
    end
end

function forward(l::SoftMaxCrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local N = size(Y, 1)
    if  size(l.last_input, 1) != N || size(l.last_output, 1) != N
        l.last_loss   = Array{Float64}(size(Y))
        l.last_input  = Array{Float64}(size(Y))
        l.last_output = Array{Float64}(size(Y))
        l.loss        = Array{Float64}(N)
        l.pred        = Array{Int64}(N)
        l.label       = Array{Int64}(N)
        l.iter        = 1:N
    end

    # l.last_input = Y
    broadcast!(-, l.last_input, Y, maximum(Y, 2))
    broadcast!(-, l.last_output, log(sum(exp(l.last_input),2)), l.last_input)
    map!(x -> convert(Int64, x) + 1,        l.label, label)
    map!(i -> l.last_output[i, l.label[i]], l.loss,  l.iter)
    map!(i -> findmax(Y[i,:])[2] - 1,       l.pred,  l.iter)

    return l.loss, l.pred
end

function backward(l::SoftMaxCrossEntropyLoss, label::Array{Float64, 2};kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local N = size(l.last_input, 1)
    if size(l.target,1) != N
        l.target    = zeros(size(l.last_input))
        l.last_loss = Array{Float64}(size(l.last_input))
    end

    # local TAR = zeros(size(l.last_input))
    # for i = 1:N
    fill!(l.target, 0)
    for i = l.iter
        l.target[i,l.label[i]] = 1
    end

    # local Y = l.last_input
    # Y = exp(broadcast(+, Y, - maximum(Y,2)))
    l.last_loss = exp(l.last_input)
    broadcast!(/, l.last_loss, l.last_loss, sum(l.last_loss,2))
    broadcast!(-, l.last_loss, l.last_loss, l.target)
    # return Y .- TAR
    return l.last_loss
end

# l = SoftMaxCrossEntropyLoss()
# X = rand(500, 10) #input size 784, batch size 500
# L = map(x -> ceil(x), rand(500, 1))
#
# println("First time (compiling...)")
# @time forward(l,X,L)
# @time backward(l,L)
#
# println("Second time ...")
# @time forward(l,X,L)
# @time backward(l,L)
#
#
# println("Third time (profiling...)")
# @time begin
#   for i = 1:1000
#     forward(l,X,L)
#   end
# end
# @time begin
#   for i = 1:1000
#     backward(l,L)
#   end
# end

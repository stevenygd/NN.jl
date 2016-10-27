include("Criteria.jl")

verbose = false

type CrossEntropyLoss <: LossCriteria
    last_loss   :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    function CrossEntropyLoss()
        return new(Float64[], Float64[])
    end
end    

function forward(l::CrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 1})
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
    if verbose
        println("Loss:$(loss); y=$(y)")
        println("output=$(l.last_output) class=$(class)")   
    end
    # println("Loss layer:$(loss)")
    return loss
end

function backward(l::CrossEntropyLoss, label::Array{Float64, 1})
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local N = size(l.last_input)[1]
    local T = zeros(size(l.last_input))
    for i = 1:N
        T[i, convert(Int64,label[1]) + 1] = 1.
        @assert sum(T[i,:]) == 1 && minimum(T[i,:]) >= 0
    end

    local Y = exp(broadcast(+, l.last_input, - maximum(l.last_input, 2)))
    @assert size(Y) == size(l.last_input)
    Y = broadcast(/, Y, sum(Y, 2))
    local DLDY = Y .- T
    if verbose
        println("dldy = $(dldy)")
    end
    return DLDY 
end
l = CrossEntropyLoss()
lbl = map(x -> convert(Float64, x), rand(0:9,10))
println(size(lbl))
println(lbl)
println(forward(l, rand(5,10), lbl))
println(backward(l, lbl))

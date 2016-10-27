include("Criteria.jl")
type CrossEntropyLoss <: LossCriteria
    last_loss   :: Array{Float64}
    last_input  :: Array{Float64}
    last_output :: Array{Float64}
    function CrossEntropyLoss()
        return new(Float64[], Float64[])
    end
end    

function forward(l::CrossEntropyLoss, y::Array{Float64,1}, label::Array{Float64, 1})
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local class = convert(Int64,label[1]) + 1
    # y = map(x -> maximum([-exp(10), x]), y - maximum(y))
    l.last_input = y
    y = y - maximum(y)
    l.last_output = -y + log(sum(exp(y)))
    local loss = l.last_output[class] 
    println("Loss:$(loss); y=$(y)")
    println("output=$(l.last_output) class=$(class)")
    if loss > exp(10)
        loss = exp(10)
    end
    # println("Loss layer:$(loss)")
    return loss
end

function backward(l::CrossEntropyLoss, x::Array{Float64,1}, label::Array{Float64, 1})
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local class = convert(Int64,label[1]) + 1
    local t = zeros(length(x))
    t[class] = 1.
    @assert sum(t) == 1 && minimum(t) >= 0
    local dldy = l.last_output - t
    println("dldy = $(dldy)")
    return dldy
end
l = CrossEntropyLoss()
println(forward(l, [1.,2.,0.], [2.]))
println(backward(l, [1.,2.,0.], [2.]))

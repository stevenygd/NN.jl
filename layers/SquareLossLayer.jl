include("Criteria.jl")

type SquareLossLayer <: Layer
    last_input :: Array{Float64}
    last_output :: Array{Float64}
    function SquareLossLayer()
        return new(Float64[], Float64[])
    end
end

function forward(l::SquareLossLayer, Y::Array{Float64,2}, t::Array{Float64, 2})
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    @return:
    loss: 
    //TODO:
    pred: naive prediction that if greater than 0 then 1 else 0
    
    """
    N = size(Y)[1]
    l.last_input = Y
    
    local temp = Y - t
    local loss = temp' * temp / N
    local pred = map(x -> (x > 0)?1:0, Y)    
    return loss, pred
    
end

function backward(l::SquareLossLayer, t::Array{Float64, 2})
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    local N = convert(Float64, size(l.last_input)[1])
    local DLDY = 2 * (l.last_input - t) / N
    return DLDY
end

l = SquareLossLayer()
Y = zeros(4,1)
Y[1,:] = 0
Y[2,:] = 1
Y[3,:] = 3
Y[4,:] = 5
t = zeros(4,1)
t[1,:] = 0
t[2,:] = 1
t[3,:] = 1
t[4,:] = 0

loss, pred = forward(l, Y, t)
println((loss, pred))
println(backward(l, t))


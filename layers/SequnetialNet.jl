abstract NN
type SequentialNet <: NN
    layers :: Array{Layer}
    lossfn :: LossCriteria
    function SequentialNet(layers::Array{Layer}, lossfn::LossCriteria)
        return new(layers, lossfn)
    end
end

function forward(net::SequentialNet, x::Array{Float64}, label::Array)
    local inp = x
    for i = 1:length(net.layers)
        inp = forward(net.layers[i], inp)
    end
    local loss = forward(net.lossfn, inp, label)
    println("Network bastract loss:$(loss)")
    return loss
end

function backward(net::SequentialNet, label)
    local dldy = backward(net.lossfn, net.layers[end].last_output, label)
    for i = length(net.layers):-1:1
        dldy = backward(net.layers[i], dldy)
    end
    return dldy
end

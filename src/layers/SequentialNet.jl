type SequentialNet <: ANN
    layers :: Array{Layer}
    lossfn :: LossCriteria
    function SequentialNet(layers::Array{Layer}, lossfn::LossCriteria)
        # Initialize all the layers
        config = Dict{String, Any}()
        @assert isa(layers[1], DataLayer)
        for i = 2:length(layers)
            init(layers[i], layers[i-1], config)
        end
        init(lossfn, layers[end], config)
        return new(layers, lossfn)
    end
end

 # TODO could remove the local
 # TODO inp could be updated in-places
 # TODO define inplace operator
function forward(net::SequentialNet, x::Array{Float64}, label::Array{Int,2}; kwargs...)
    local inp = x
    for i = 1:length(net.layers)
        inp = forward(net.layers[i], inp; kwargs...)
    end
    loss, pred = forward(net.lossfn, inp, label; kwargs...)
    # println("Network bastract loss:$(loss)")
    return loss, pred
end

function backward(net::SequentialNet, label)
    local dldy = backward(net.lossfn, label)
    for i = length(net.layers):-1:1
        dldy = backward(net.layers[i], dldy)
    end
    return dldy
end

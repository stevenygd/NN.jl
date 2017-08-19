type ComputationGraph <: ANN
    sorted_layers :: Array{Layer}

    function ComputationGraph(root::Layer)
        # Initialize all the layers
        config = Dict{String, Any}()
        #TODO
        return new(layers, lossfn)
    end
end

function forward(graph::ComputationGraph, x::Array{Float64}, label::Array; kwargs...)
    #TODO
end

function backward(graph::ComputationGraph, label)
    #TODO
end

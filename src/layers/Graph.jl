type Graph <: ANN
    layer :: Layer
    forward_order :: Array{Layer}

    function Graph(layer::Layer)
        # Initialize all the layers
        graph = new(layers)
        top_sort()
        return graph
    end
end

function top_sort(graph::Graph)
    l = graph.layer
    visited = Set([l])
    frontier = [l]
    while length(l) > 0
        layer = frontier[0]
        
    end
    return Layer[] # return an array of layer
end

function forward(graph::Graph)
    return Float64[]
end

function backward(graph::Graph)
    return Float64[]
end

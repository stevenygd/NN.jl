type Graph <: ANN
    layer :: Layer
    forward_order :: Array{Layer}
    visited :: Set{Layer}

    function Graph(layer::Layer)
        # Initialize all the layers
        graph = new(layer)
        graph.visited = Set{Layer}([graph.layer])
        top_sort(graph, graph.layer)
        return graph
    end
end

function top_sort(graph::Graph, layer::Layer)
    push!(graph.visited, layer)
    for i=1:length(layer.parents)
        l = layer.parent[i]
        if !in(l, graph.visited)
            top_sort(graph, l)
        end
        append!(graph.forward_order, l)
    end
end

function forward(graph::Graph)
    return Float64[]
end

function backward(graph::Graph)
    return Float64[]
end

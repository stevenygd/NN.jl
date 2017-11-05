type Graph <: ANN
    forward_order :: Array{Layer} # list of pointers to layers in order of calling forward
    input_layers :: Array{DataLayer} # list of points to input layers in the graph

    function Graph(layer::Layer)
        graph = new(Layer[], Layer[])
        top_sort(graph, layer, Set{Layer}([layer]))
        # for i=1:length(graph.forward_order)
        #     if isa(graph.forward_order[i], DataLayer)
        #         push!(graph.input_layers, graph.forward_order[i])
        #     end
        # end
        return graph
    end
end

function top_sort(graph::Graph, layer::Layer, visited::Set{Layer})
    push!(visited, layer)
    for i=1:length(layer.parents)
        l = layer.parents[i]
        if !in(l, visited)
            top_sort(graph, l, visited)
        end
    end
	if isa(layer, DataLayer)
        push!(graph.input_layers, layer)
    else
        push!(graph.forward_order, layer)
    end
end

function forward(graph::Graph, xs::Dict{String, Array{Float64}}; kwargs...)
    for x in keys(xs)
        for i=1:length(graph.input_layers)
            input = graph.input_layers[i]
            if input.tag==x
                forward(input, xs[x];kwargs...)
            end
        end
    end
    forward_the_rest(graph; kwargs...)
end

function forward_the_rest(graph::Graph; kwargs...)
    for i=1:length(graph.forward_order)
        layer = graph.forward_order[i]
        forward(layer; kwargs...)
    end
end

function backward(graph::Graph)
    return Float64[]
end

function inference(graph::Graph, layer::Layer, xs::Dict{String,Array{Float64}})
    return Float64[]
end

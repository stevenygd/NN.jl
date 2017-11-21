type Graph <: ANN
    forward_order :: Array{Layer} # list of pointers to layers in order of calling forward
    input_layers :: Dict{String, DataLayer} # dictionary of input layer tags to input layers

    function Graph(layer::Layer)
        graph = new(Layer[], Dict{String,DataLayer}())
        top_sort(graph, layer, Set{Layer}([layer]))
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
        graph.input_layers[layer.tag] = layer
    else
        push!(graph.forward_order, layer)
    end
end

function forward(graph::Graph, xs::Dict{String, Array{Float64}}; kwargs...)
    for x in keys(xs)
        forward(graph.input_layers[x], xs[x];kwargs...)
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
    for l ∈ graph.forward_order
        backward(l)
    end
end

function inference(graph::Graph, layer::Layer, xs::Dict{String,Array{Float64}})
    return Float64[]
end

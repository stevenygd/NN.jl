type Graph <: ANN
    layer :: Layer #topmost layer in the grpah, the child of all other layers
    forward_order :: Array{Layer} # list of pointers to layers in order of calling forward
    input_layers :: Array{Layer} # list of points to input layers in the graph

    function Graph(layer::Layer)
        graph = new(layer, Layer[], Layer[])
        top_sort(graph, graph.layer, Set{Layer}([graph.layer]))
        for i=1:length(graph.forward_order)
            if isa(graph.forward_order[i], DataLayer)
                push!(graph.input_layers, graph.forward_order[i])
            end
        end
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
	if !isa(layer, DataLayer)
        push!(graph.forward_order, layer)
    end
end

function forward(graph::Graph, x::Array{Float64}, label::Array; kwargs...)
    for i=1:length(graph.input_layers)
        forward(graph.input_layers[i], x)
    end
    forward_the_rest(graph, label)
end

function forward(graph::Graph, xs::Dict{String, Array{Float64}}, label::Array; kwargs...)
    for x in keys(xs)
        for i=1:length(graph.input_layers)
            input = graph.input_layers[i]
            if input.tag==x
                forward(input, xs.get(x))
            end
        end
    end
    forward_the_rest(graph, label)
end

function forward_the_rest(graph::Graph, label::Array; kwargs...)
    for i=1:length(graph.forward_order)
        layer = graph.forward_order[i]
        if isa(layer, LossCriteria)
            loss, pred = forward(layer, label)
        else
            forward(layer)
        end
    end
    return loss, pred
end

function backward(graph::Graph)
    return Float64[]
end

function inference(graph::Graph, layer::Layer, xs::Dict{String,Array{Float64}})
    return Float64[]
end

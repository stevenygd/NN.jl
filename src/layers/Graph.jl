include("LayerBase.jl")
type Graph <: ANN
    layer :: Layer
    forward_order :: Array{Layer}
    visited :: Set{Layer}
    input_layers :: Array{Layer}

    function Graph(layer::Layer)
        graph = new(layer, Layer[], Set{Layer}(), Layer[])
        graph.visited = Set{Layer}([graph.layer])
        top_sort(graph, graph.layer)
        for i=1:length(graph.forward_order)
            if isa(graph.forward_order[i], DataLayer)
                push!(graph.input_layers, graph.forward_order[i])
            end
        end
        return graph
    end
end

function top_sort(graph::Graph, layer::Layer)
    push!(graph.visited, layer)
    for i=1:length(layer.parents)
        l = layer.parents[i]
        if !in(l, graph.visited)
            top_sort(graph, l)
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
end

function forward(graph::Graph, xs::Dict{String, Array{Float64}}, label::Array; kwargs...)
    for x in keys(xs)
        for i=1:length(graph.input_layers)
            input = input_layers[i]
            if input.tag==x
                forward(input, xs.get(x))
            end
        end
    end

end

function backward(graph::Graph)
    return Float64[]
end

function inference(graph::Graph, layer::Layer, xs::Dict{String,Array{Float64}})
    return Float64[]
end

include("../src/layers/InputLayer.jl")
include("../src/layers/CaffeConvLayer.jl")
include("../src/layers/ReLu.jl")
include("../src/layers/MaxPoolingLayer.jl")
include("../src/layers/FlattenLayer.jl")
include("../src/layers/DenseLayer.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/Graph.jl")
include("../src/layers/SequentialNet.jl")
using Base.Test

function GraphTopsortTest(graph, expected)

    # compare(l::layer, l2::layer)?
    for i=1:length(expected)
        @test graph[i] == expected[i]
    end
end

batch_size = 10
config = Dict{String, Any}()
l0 = InputLayer(nothing, (28,28,1,batch_size), config)
l1 = CaffeConvLayer(l0, 32,(5,5), config)
l2 = ReLu(l1, config)
l3 = MaxPoolingLayer(l2,(2,2), config)
l4 = FlattenLayer(l3, config)
l5 = DenseLayer(l4, 10, config)
l6 = SoftMaxCrossEntropyLoss(l5,config)
graph1 = Graph(l6)
expected = [l1, l2, l3, l4, l5, l6]
# println(typeof(expected[1]))
# println(typeof(graph1[1]))
GraphTopsortTest(graph1.forward_order, expected)
println("Basic topsort test passed")

function GraphForwardTest(graph::Graph, layer::Layer, net::SequentialNet, x::Array{Float64}, label::Array{Float64})
    forward(net, x, label)
    actual = forward(graph, x, label)
    @test layer.y == actual[1]
end

function build_cnn()
    layers = Layer[
        InputLayer((28,28,1,batch_size)),
        CaffeConvLayer(32,(5,5)),
        ReLu(),
        MaxPoolingLayer((2,2)),
        FlattenLayer(),
        DenseLayer(10)
    ]
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

net1 = build_cnn()
GraphForwardTest(graph1, l6, net1, rand(28,28,1,10), rand(10,10))

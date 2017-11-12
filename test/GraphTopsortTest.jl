include("../src/layers/InputLayer.jl")
include("../src/layers/CaffeConvLayer.jl")
include("../src/layers/ReLu.jl")
include("../src/layers/MaxPoolingLayer.jl")
include("../src/layers/AdditionLayer.jl")
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
l0 = InputLayer((28,28,1,batch_size), config)
l1 = CaffeConvLayer(l0, 32,(5,5), config)
l2 = ReLu(l1, config)
l3 = MaxPoolingLayer(l2,(2,2), config)
l4 = FlattenLayer(l3, config)
l5 = DenseLayer(l4, 10, config)
l6 = SoftMaxCrossEntropyLoss(l5,config)
graph1 = Graph(l6)
expected = [l1, l2, l3, l4, l5, l6]
GraphTopsortTest(graph1.forward_order, expected)
println("Basic topsort test passed")

l0_2 = InputLayer((500, 784), config)
l1_2 = DenseLayer(l0_2, 10, config)
l2_2 = DenseLayer(l1_2, 10, config)
l3_2 = DenseLayer(l1_2, 10, config)
l4_2 = AdditionLayer([l3_2, l2_2],config)
graph2 = Graph(l4_2)
@test findfirst(graph2.forward_order, l1_2) < findfirst(graph2.forward_order, l2_2)
@test findfirst(graph2.forward_order, l1_2) < findfirst(graph2.forward_order, l3_2)
@test findfirst(graph2.forward_order, l1_2) < findfirst(graph2.forward_order, l4_2)
@test findfirst(graph2.forward_order, l2_2) < findfirst(graph2.forward_order, l4_2)
@test findfirst(graph2.forward_order, l3_2) < findfirst(graph2.forward_order, l4_2)
println("Basic asequential topology test passed")

l0_3 = InputLayer((500, 784), config)
l1_3 = DenseLayer(l0_3, 10, config)
l2_3 = DenseLayer(l1_3, 10, config)
l3_3 = DenseLayer(l1_3, 10, config)
l4_3 = AdditionLayer([l3_3, l2_3], config)
l5_3 = AdditionLayer([l4_3, l3_3], config)
graph3 = Graph(l5_3)
@test findfirst(graph3.forward_order, l1_3) < findfirst(graph3.forward_order, l2_3)
@test findfirst(graph3.forward_order, l1_3) < findfirst(graph3.forward_order, l3_3)
@test findfirst(graph3.forward_order, l1_3) < findfirst(graph3.forward_order, l4_3)
@test findfirst(graph3.forward_order, l2_3) < findfirst(graph3.forward_order, l4_3)
@test findfirst(graph3.forward_order, l3_3) < findfirst(graph3.forward_order, l4_3)
@test findfirst(graph3.forward_order, l4_3) < findfirst(graph3.forward_order, l5_3)
@test findfirst(graph3.forward_order, l3_3) < findfirst(graph3.forward_order, l5_3)
println("More complex topology test passed")

function GraphForwardTest(graph::Graph, layer::Layer, net::SequentialNet, x::Array{Float64}, labels::Array{Float64})
    forward(net, x, labels)
	xs = Dict{String,Array{Float64}}("default"=>x, "labels" => labels)
    forward(graph, xs)
    @test layer.y == net.lossfn.y
end

l0_n = InputLayer((28,28,1,batch_size))
l1_n = CaffeConvLayer(32,(5,5))
l2_n = ReLu()
l3_n = MaxPoolingLayer((2,2))
l4_n = FlattenLayer()
l5_n = DenseLayer(10)

layers = Layer[l0_n,l1_n,l2_n,l3_n,l4_n,l5_n]
l6_n = SoftMaxCrossEntropyLoss()
net = SequentialNet(layers, l6_n)
layers = Layer[l1_n,l2_n,l3_n,l4_n,l5_n,l6_n]

x = rand(28,28,1,10)
labels = rand(10,10)
forward(net, x, labels)
xs = Dict{String,Array{Float64}}("default"=>x, "labels" => labels)
l1.kern = l1_n.kern
l5.W = l5_n.W
forward(graph1, xs)
for i=1:6
    @test expected[i].y==layers[i].y
end
GraphForwardTest(graph1, l6, net, rand(28,28,1,10), rand(10,10))
println("Basic forward test passed")

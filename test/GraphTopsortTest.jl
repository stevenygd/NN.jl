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

GraphTopsortTest(graph1.forward_order, expected)
println("Basic topsort test passed")

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
forward(graph1, xs)

@test l1.parents[1].y == l0_n.y
@test l1.x == l1_n.x
@test l1.y == l1_n.y
# for i=1:6
#     println(typeof(expected[i]))
#     println(typeof(layers[i]))
#     @test expected[i].y==layers[i].y
# end
# GraphForwardTest(graph1, l6, net, rand(28,28,1,10), rand(10,10))

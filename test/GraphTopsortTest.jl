include("../src/layers/InputLayer.jl")
include("../src/layers/CaffeConvLayer.jl")
include("../src/layers/ReLu.jl")
include("../src/layers/MaxPoolingLayer.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/Graph.jl")

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
l4 = SoftMaxCrossEntropyLoss(l3,config)
graph1 = Graph(l4)
expected = [l1, l2, l3, l4]
# println(typeof(expected[1]))
# println(typeof(graph1[1]))
GraphTopsortTest(graph1.forward_order, expected)
println("Basic test passed")

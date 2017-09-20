include("../src/NN.jl")
using Base.Test

function GraphTopsortTest(graph, order)

    # compare(l::layer, l2::layer)?
    @test graph.forward_order == order
end

l0 = InputLayer(Void, (28,28,1,batch_size))

l1 = CaffeConvLayer(l0, 32,(5,5))
l2 = ReLu(l1)
l3 = MaxPoolingLayer(l2,(2,2))
l4 = SoftMaxCrossEntropyLoss(l3)
graph1 = Graph(l4)

GraphTopsortTest(graph1, [l0, l1, l2, l3, l4])

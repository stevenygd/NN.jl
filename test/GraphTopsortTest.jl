include("src/NN.jl")

function build_graph()
    l0 = InputLayer(Void, (28,28,1,batch_size))

    l1 = CaffeConvLayer(l0, 32,(5,5))
    l2 = ReLu(l1)
    l3 = MaxPoolingLayer(l2, (2,2))

    l4 = SoftMaxCrossEntropyLoss(l3)

    graph = Graph(l4)
    return graph
end

function test GraphTopsortTest(graph, order)
    @test graph.forward_order == order
end

graph = build_graph()
println(graph.forward_order)


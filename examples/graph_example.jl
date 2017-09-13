include("src/NN.jl")
include("util/datasets.jl")

batch_size = 500

function build_graph()
    l = InputLayer(l, (28,28,1,batch_size)),

    l = CaffeConvLayer(l, 32,(5,5)),
    l = ReLu(),
    l = MaxPoolingLayer(l, (2,2)),

    l = CaffeConvLayer(l, 32,(5,5)),
    l = ReLu(l),
    l = MaxPoolingLayer(l, (2,2)),

    l = FlattenLayer(l),

    l = DenseLayer(l, 256),
    l = ReLu(l),

    l = DropoutLayer(l, 0.5),
    l = DenseLayer(l, 10)

    graph = Graph(l)
    return graph
end

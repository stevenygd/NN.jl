include("src/NN.jl")
include("util/datasets.jl")

using NN
using PyPlot
using IProfile

batch_size = 500

function build_model()
    layerX = InputLayer((28,28,1,batch_size))
    layerY = InputLayer((batch_size,10))
    c1 = CaffeConv(layerX, 32, (5,5);padding=4)
    r1 = Relu(c1);
    m1 = MaxPoolingLayer(r1, (2,2))
    c2 = CaffeConv(m1, 32, (5,5);padding=4)
    r2 = Relu(c2);
    m2 = MaxPoolingLayer(r2, (2,2))
    f = Flatten(m2)

    return layerX, layerY, Graph(l6)
end

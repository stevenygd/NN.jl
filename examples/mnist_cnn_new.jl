include("src/NN.jl")
include("util/datasets.jl")

using NN
using PyPlot
using IProfile

batch_size = 500

function build_model()
    layerX = InputLayer((28,28,1,batch_size))
    layerY = InputLayer((batch_size,10))
    l1 = CaffeConv(layerX, 32, (5,5);padding=4)
    l2 = Relu(l1);
    
    return layerX, layerY, Graph(l6)
end

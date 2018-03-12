# include("../src/layers/LayerBase.jl")
# include("../src/layers/InputLayer.jl")
# include("../src/layers/FullyConnectedFixed.jl")
# include("../src/layers/SoftMaxCrossEntropy.jl")
# include("../src/layers/Graph.jl")
include("../src/NN.jl")
using NN

import Calculus: check_gradient
using Base.Test

tol = 1e10

l1 = InputLayer((10, 100))
l2 = FullyConnectedFixed(l1, 1000)

function testFullyConnectedOneVector(w, b, x, y, dldy, dldx, gw, gb)
    out_size = size(w, 1)
    l1 = InputLayer(size(x))
    l2 = FullyConnectedFixed(l1, size(w, 2))
    l4 = InputLayer((out_size, out_size))
    l3 = SoftMaxCrossEntropyLoss(l1,l4)
    g  = Graph(l3)
    xs = Dict{Layer,Array{Float64}}(l1=>x, l4 => zeros(out_size, out_size))
    # Testing forwarding
    W = [w; b;]
    setParam!(l2, Array[W])
    n = size(x,2) # x is two dimensional
    forward(g, xs)
end

include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/Graph.jl")
import Calculus: check_gradient
using Base.Test


function testSoftMaxCrossEntropyOneVector(x, labels, p, loss, dldx; alpha = 1.)
    xs = Dict{String,Array{Float64}}("default"=>x, "labels" => labels)
    l1 = InputLayer(size(x))
    l2 = SoftMaxCrossEntropyLoss(l1)
    g  = Graph(l2)
    # Testing forwarding
    forward(g, xs)
    @test l2.loss ≈ loss

    # Testing back propagation
    backward(g)
    @test l2.base.dldx[l1.base.id] ≈ dldx
end

# First Test
println("Unit test 1...")
x = [1. 2. 3.; -1. -2. -3.]
s = e^(-2) + e^(-1) + 1.
y = [0. 0. 1.; 0. 1. 0.]
p = [2 ; 0]
loss = -log.([1./s   ;   e^(-1)/s])

dldx = [ e^(-2)/s e^(-1)/s (1./s - 1); 1./s (e^(-1)/s-1) e^(-2)/s]
testSoftMaxCrossEntropyOneVector(x, y, p, loss, dldx)
println("test 1 passed.\n")

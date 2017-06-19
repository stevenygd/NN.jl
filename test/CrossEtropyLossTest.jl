include("../src/layers/LayerBase.jl")
include("../src/layers/CrossEntropyLoss.jl")
import Calculus: check_gradient
using Base.Test

l = CrossEntropyLoss()

function beforeTest(l)
    init(l, nothing, Dict{String, Any}("batch_size" => 2, "input_size" => [3]))
end

function basicTest(l, x, label, predict, loss, dldx; alpha = 1.)
    beforeTest(l)

    # Testing forwarding
    fl, fp = forward(l,x,label)
    @test_approx_eq fl loss
    @test_approx_eq fp predict

    # Testing back propagation
    @test_approx_eq backward(l,label) dldx
end

x = [.7 .2 .1 ; .03 .29 .68]
pred = zeros(2,1)
pred[:,1] = [0;2]
loss = zeros(2,1)
loss[:,1] = [-log(.7);-log(0.68)]
label=[1 0 0; 0 0 1]

basicTest(l,x,pred,pred, loss, -label./x)
println("Basic test passed")

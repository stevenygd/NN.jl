include("../src/layers/LayerBase.jl")
include("../src/layers/InputLayer.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
import Calculus: check_gradient
using Base.Test

function beforeTest()
    return InputLayer((1,2))
end


function testInputLayerOneVector(x, y, dldy, dldx)
    l = beforeTest()
    l2 = SoftMaxCrossEntropyLoss(l)

    # Testing forwarding
    @test forward(l,x) == y
    @test l.x == x
    @test l.y == y

    #Testing back propagation
    l2.dldx[l.id] = dldy
    @test backward(l) == dldx
    @test l.dldy == dldy
end

# First Test
println("Unit test 1...")
x = [1. 1.;]
y = [1. 1.;]
dldy = [0. 0.;]
dldx = [0. 0.;]
testInputLayerOneVector(x, y, dldy, dldx)
println("test 1 passed.\n")

# Second Test
println("Unit test 2...")
x2 = [2. 3.;]
y2 = [2. 3.;]
dldy2 = [0. 1;]
dldx2 = [0. 1;]
testInputLayerOneVector(x2, y2, dldy2, dldx2)
println("test 2 passed.\n")

# Third test
println("Unit test 3...")
x  = [6. 5;]
y  = [6. 5;]
dldy = [1. 2;]
dldx = [1. 2;]
testInputLayerOneVector(x, y, dldy, dldx)
println("test 3 passed.\n")

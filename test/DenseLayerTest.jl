include("../src/layers/LayerBase.jl")
include("../src/layers/DenseLayer.jl")
import Calculus: check_gradient
using Base.Test

l = DenseLayer(2)
function beforeTest(l)
    init(l, nothing, Dict{String, Any}("batch_size" => 1, "input_size" => [2]))
end


function testDenseLayerOneVector(w, b, x, y, dldy, dldx, gw, gb)
    beforeTest(l)

    # Testing forwarding
    W = [w; b;]
    setParam!(l, W)
    n = size(x,2) # x is two dimensional
    @test forward(l,x) == y
    @test l.x[1:n] == x[1,:]
    @test l.y == y

    #Testing back propagation
    @test backward(l,dldy) == dldx
    @test l.dldy == dldy
    @test l.dldx[1:n] == dldx[1,:]
    @test getGradient(l)[1:end-1, :]' == gw
    @test getGradient(l)[end, :]' == gb
end

# First Test
println("Unit test 1...")
w = [1. 1.;
     1. 1.;]
b = [0. 0;]
x = [1. 1.;]
y = [2. 2.;]
dldy = [0. 0.;]
dldx = [0. 0.;]
gw = [0. 0.;0. 0.;]
gb = [0. 0.]
testDenseLayerOneVector(w, b, x, y, dldy, dldx, gw, gb)
println("test 1 passed.\n")

# Second Test
println("Unit test 2...")
x2 = [2. 3.;]
y2 = [5. 5.;]
dldy2 = [0. 1;]
dldx2 = [1. 1;]
gw2 = [0. 0.;2. 3.; ]
gb2 = [0. 1;]  # bias is dldy
testDenseLayerOneVector(w, b, x2, y2, dldy2, dldx2, gw2, gb2)
println("test 2 passed.\n")

# Third test
println("Unit test 3...")
w2 = [2. 3.; 3. 2.]
b2 = [1. 0;]
x  = [1. 1;]
y  = [6. 5;]
dldy = [1. 2;]
dldx = [8. 7;]
gw2 = [1. 1; 2. 2;]
gb2 = [1. 2;]
testDenseLayerOneVector(w2, b2, x, y, dldy, dldx, gw2, gb2)
println("test 3 passed.\n")

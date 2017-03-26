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
    setParam!(l, Array[W])
    n = size(x,2) # x is two dimensional
    @test forward(l,x) == y
    @test l.x[1:n] == x[1,:]
    @test l.y == y

    #Testing back propagation
    @test backward(l,dldy) == dldx
    @test l.dldy == dldy
    @test l.dldx[1:n] == dldx[1,:]
    @test getGradient(l)[1][1:end-1, :]' == gw
    @test getGradient(l)[1][end, :]' == gb
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

include("gradient_test.jl")

bsize= 1
l = DenseLayer(30)
in_size = 50
init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => (in_size)))
X = rand(bsize, in_size)
Y = ones(size(forward(l, X)))
backward(l,Y)
g  = getGradient(l)[1]
w = copy(l.W)
function f_w1(w)
    setParam!(l, Array[w])
    return sum(forward(l,X))
end
anl_g, err = gradient_check(f_w1, g, w)
println("Relative error: $(err) $(mean(abs(anl_g))) $(mean(abs(g)))")
@test_approx_eq_eps anl_g g 1e-4
@test_approx_eq_eps err 0. 1e-4
println("[PASS] gradient check test 1.")

include("../src/layers/InputLayer.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/SequentialNet.jl")
batch_size = 500
inp_size    = 30
out_size    = 10
function build_mlp()
    l = DenseLayer(out_size)
    layers = Layer[
        InputLayer((batch_size,inp_size)), l
    ]
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return l, net
end
l, net = build_mlp()
X = rand(batch_size, inp_size)
Y = zeros(Int, batch_size, out_size)
Y[1:batch_size, rand(1:10, batch_size)] = 1
function f_w2(w)
    setParam!(l, Array[w])
    loss, _ = forward(net, X, Y)
    return sum(loss)
end
w = l.W
forward(net, X, Y)
backward(net, Y)
g = getGradient(l)[1]
anl_g, err = gradient_check(f_w2, g, w)
println("Relative error: $(err) $(mean(abs(anl_g))) $(mean(abs(g)))")
@test_approx_eq_eps anl_g g 1e-4
@test_approx_eq_eps err 0. 1e-4
println("[PASS] gradient check test 2.")

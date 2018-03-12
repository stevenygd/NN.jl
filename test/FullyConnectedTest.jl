include("../src/layers/LayerBase.jl")
include("../src/layers/InputLayer.jl")
include("../src/layers/FullyConnected.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/Graph.jl")

import Calculus: check_gradient
using Base.Test

tol = 1e10

function testFullyConnectedOneVector(w, b, x, y, dldy, dldx, gw, gb)
    out_size = size(w, 1)
    l1 = InputLayer(size(x))
    l2 = FullyConnected(l1, size(w, 2))
    l4 = InputLayer((out_size, out_size))
    l3 = SoftMaxCrossEntropyLoss(l1,l4)
    g  = Graph(l3)
    xs = Dict{Layer,Array{Float64}}(l1=>x, l4 => zeros(out_size, out_size))
    # Testing forwarding
    W = [w; b;]
    setParam!(l2, Array[W])
    n = size(x,2) # x is two dimensional
    forward(g, xs)
    @test l2.x[1:n] ≈ x[1,:] atol=tol
    @test l2.base.y ≈ y atol=tol

    #Testing back propagation
    l3.base.dldx[l2.base.id] = dldy
    backward(l2)
    @test l2.dldy ≈ dldy atol=tol
    @test l2.base.dldx[l1.base.id][1:n] ≈ dldx[1,:] atol=tol
    @test getGradient(l2)[1][1:end-1, :]' ≈ gw atol=tol
    @test getGradient(l2)[1][end, :]' ≈ gb atol=tol
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
testFullyConnectedOneVector(w, b, x, y, dldy, dldx, gw, gb)
println("test 1 passed.\n")

# Second Test
println("Unit test 2...")
x2 = [2. 3.;]
y2 = [5. 5.;]
dldy2 = [0. 1;]
dldx2 = [1. 1;]
gw2 = [0. 0.;2. 3.; ]
gb2 = [0. 1;]  # bias is dldy
testFullyConnectedOneVector(w, b, x2, y2, dldy2, dldx2, gw2, gb2)
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
testFullyConnectedOneVector(w2, b2, x, y, dldy, dldx, gw2, gb2)
println("test 3 passed.\n")

include("gradient_check.jl")

bsize= 1
in_size = 50
out_size = 30
l1 = InputLayer((bsize, in_size))
l2 = FullyConnected(l1, out_size)
X = rand(bsize, in_size)
Y = ones(forward(l2, X))
backward(l2, Y)
g = getGradient(l2)[1]
w = copy(l2.W)
function f_w1(w)
    setParam!(l2, Array[w])
    return sum(forward(l2, X))
end
anl_g, err = gradient_check(f_w1, g, w)
println("Relative error: $(err) $(mean(abs.(anl_g))) $(mean(abs.(g)))")
@test anl_g ≈ g atol=1e-4
@test err ≈ 0. atol=1e-4
println("[PASS] gradient check test 1.")

batch_size  = 500
inp_size    = 30
out_size    = 10
function build_mlp()
    l1 = InputLayer((batch_size, inp_size))
    l2 = FullyConnected(l1, out_size)
    l3 = SoftMaxCrossEntropyLoss(l2)
    graph = Graph(l3)
    return l2, graph
end
l2, graph = build_mlp()
X = rand(batch_size, inp_size)
Y = zeros(batch_size, out_size)
Y[1:batch_size, rand(1:10, batch_size)] = 1
function f_w2(w)
    setParam!(l2, Array[w])
    loss, _ = forward(graph, Dict("default"=>X, "labels"=>Y))
    return sum(loss)
end
w = l2.W
forward(graph, Dict("default"=>X, "labels"=>Y))
backward(graph)
g = getGradient(l2)[1]
anl_g, err = gradient_check(f_w2, g, w)
println("Relative error: $(err) $(mean(abs.(anl_g))) $(mean(abs.(g)))")
@test anl_g ≈ g atol=1e-4
@test err ≈ 0. atol=1e-4
println("[PASS] gradient check test 2.")

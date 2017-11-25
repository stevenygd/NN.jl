include("../src/layers/LayerBase.jl")
include("../src/layers/DropoutLayer.jl")
include("../src/layers/InputLayer.jl")
using Base.Test

function testDropout(x, y, dldx, dldy)
    l0 = InputLayer(size(x))
    l  = DropoutLayer(l0, 0.5)
    println("successfully create DropoutLayer")
end

# Unit Test
x = [1. 0. 0.5;]
y = [1. 0. 0.5;]
dldy = [1. 2. 3.;]
dldx = [1. 0. 3.;]
testDropout(x, y, dldy, dldx)

# Gradient Check
bsize= 1
in_size = 50
out_size = 30
l1 = InputLayer((bsize, in_size))
l2 = DropoutLayer(l1, 0.5)
X = rand(bsize, in_size)
Y = ones(forward(l2, X))
backward(l2, Y)
println("successfully run forward and backward on DropoutLayer")

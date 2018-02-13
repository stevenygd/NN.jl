include("../src/layers/LayerBase.jl")
include("../src/layers/InputLayer.jl")
include("../src/layers/Flatten.jl")
import Calculus: check_gradient
using Base.Test

function testFlatten(x, y, dldy, dldx)
    l0 = InputLayer(size(x))
    l  = Flatten(l0)

    # Testing forwarding
    forward(l0, x)
    @test forward(l) == y
    @test l.x == x
    @test l.base.y == y

    #Testing back propagations
    @test backward(l, dldy) == dldx
    @test l.dldy == dldy
    @test l.base.dldx[l0.base.id] == dldx
end

println("Unit test 1...")
x = rand((5,5,3,1))
dy = reshape(x, (75, 1))
y = permutedims(dy, [2,1])
dldy = rand(1,75)
dldx = reshape(permutedims(dldy, [2,1]), size(x))
testFlatten(x, y, dldy, dldx)
println("test 1 passed.\n")

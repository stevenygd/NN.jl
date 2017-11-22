include("../src/layers/LayerBase.jl")
include("../src/layers/Graph.jl")
include("../src/layers/AdditionLayer.jl")
include("../src/layers/InputLayer.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
using Base.Test

function AdditionTest(xs, y, dldy)
    input1 = InputLayer((1, 2))
    input2 = InputLayer((1, 2))
    l      = AdditionLayer([input1, input2])
    l2     = SoftMaxCrossEntropyLoss(l)
    forward(input1, xs[1])
    forward(input2, xs[2])
    forward(l)
    l2.base.dldx[l.base.id] = dldy
    backward(l)

    @test l.base.y == y
    @test l.base.dldx[input1.base.id] == dldx1
    @test l.base.dldx[input2.base.id] == dldx2
end

xs = Array{Float64}[]
x1 = [1. 2.;]
x2 = [1. 1.;]
push!(xs, x1)
push!(xs, x2)

y = [2. 3.;]
dldy = [3. 4.;]
dldx1 = [3. 4.;]
dldx2 = [3. 4.;]
AdditionTest(xs, y, dldy)
println("AdditionLayer Test passed")

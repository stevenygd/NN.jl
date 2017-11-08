include("../src/layers/LayerBase.jl")
include("../src/layers/Graph.jl")
include("../src/layers/AdditionLayer.jl")
include("../src/layers/InputLayer.jl")
using Base.Test

function Test(xs, y, dldy)
    config = Dict{String, Any}()
    input1 = InputLayer(nothing, (1, 2), config)
    input2 = InputLayer(nothing, (1, 2), config)
    l = AdditionLayer([input1, input2], config)
    forward(input1, xs[1])
    forward(input2, xs[2])

    @test forward(l) == y
    @test backward(l, dldy) == dldx1
    @test backward(l, dldy) == dldx2
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
Test(xs, y, dldy)
println("AdditionLayer Test passed")

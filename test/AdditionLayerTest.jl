include("../src/layers/LayerBase.jl")
include("../src/layers/AdditionLayer.jl")
include("../src/layers/InputLayer.jl")
using Base.Test

function Test(xs, y, dldy)
    config = Dict{String, Any}()
    input1 = InputLayer(nothing, (1, 2), config)
    input2 = InputLayer(nothing, (1, 2), config)
    l = AdditionLayer([input1, input2], config)

    @test forward(l, xs) == y
    @test backward(l, dldy)[:, :, 1] == dldx1
    @test backward(l, dldy)[:, :, 2] == dldx2
end

xs = Array{Float64}(1, 2, 2)
x1 = [1. 2.;]
x2 = [1. 1.;]
xs[:, :, 1] = x1
xs[:, :, 2] = x2
y = [2. 3.;]
dldy = [3. 4.;]
dldx1 = [3. 4.;]
dldx2 = [3. 4.;]
Test(xs, y, dldy)

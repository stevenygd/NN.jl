include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/SoftMax.jl")
include("../src/layers/CrossEntropyLoss.jl")
using Base.Test

smaxcross = SoftMaxCrossEntropyLoss()
smax = SoftMax()
cross = CrossEntropyLoss()


function beforeTest(smaxcross, smax, cross, dict)
    init(smaxcross, nothing, dict)
    init(smax     , nothing, dict)
    init(cross    , nothing, dict)
end

function testSoftMaxCrossEntropyOneVector(smaxcross, smax, cross, x, label; alpha = 1.)
    beforeTest(smaxcross, smax, cross, Dict{String, Any}("batch_size" => 2, "input_size" => [3]))

    # Testing forwarding
    l1, p1 = forward(smaxcross,x,y)
    l2, p2 = forward(cross, forward(smax,x), label)
    @test_approx_eq l1 l2
    @test_approx_eq p1 p2
    d1 = backward(smaxcross,label)
    d2 = backward(smax, backward(cross, label))

    # Testing back propagation
    @test_approx_eq d1 d2
end

# First Test
println("Unit test 1...")
x = [1. 2. 3.; -1. -2. -3.]
y = zeros(Int64, 2,1)
y[:, 1] = [2      ;   1]
testSoftMaxCrossEntropyOneVector(smaxcross, smax, cross, x, y; alpha = 1.)
println("basic test passed.\n")

include("../src/layers/LayerBase.jl")
include("../src/layers/CrossEntropyLoss.jl")
import Calculus: check_gradient
using Base.Test

l = CrossEntropyLoss()

function beforeTest(l, x)
  m,n = size(x)
  init(l, nothing, Dict{String, Any}("batch_size" => 0, "input_size" => [n]))
end

function test(l, x, label; alpha = 1.)
    beforeTest(l,x)

    # Testing forward
    @test forward(l,x,label)[1]== sum(-label.*log(x),2)

    # Testing backward
    @test backward(l,label)==-label./x
end

println("Test 1...")
x = [.7 .2 .1 ; .03 .29 .68]
label = rand(2,3)
# loss = sum(-label.*log(x),2)
test(l,x,label)
println("Basic test passed")

println("Test 2...")
x = rand(2,10)
label = rand(2,10)
test(l,x,label)
println("Multi test passed")

println("Test 3...")
x = rand(500,100)
label = rand(500,100)
test(l,x,label)
println("Multi test 2 passed")

println("Test 4...")
x = rand(10000,1000)
label = rand(10000,1000)
test(l,x,label)
println("Final insane passed")

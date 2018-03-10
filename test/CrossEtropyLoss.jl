include("../src/layers/LayerBase.jl")
include("../src/layers/InputLayer.jl")
include("../src/layers/CrossEntropyLoss.jl")
import Calculus: check_gradient
using Base.Test

function test(x, label; alpha = 1.)
  m,n = size(x)
  input = InputLayer((m,n))
  l = CrossEntropyLoss(Dict{String, Any}("batch_size" => 0, "input_size" => n),input)

  # Testing forward
  @test forward(l,x,label)[1]== sum(-label.*log.(x),2)

  # Testing backward
  @test backward(l,label)==-label./x
end

println("Test 1...")
x = [.7 .2 .1 ; .03 .29 .68]
label = rand(2,3)

test(x,label)
println("Basic test passed")

println("Test 2...")
x = rand(2,10)
label = rand(2,10)
test(x,label)
println("Multi test passed")

println("Test 3...")
x = rand(500,100)
label = rand(500,100)
test(x,label)
println("Multi test 2 passed")

println("Test 4...")
x = rand(10000,1000)
label = rand(10000,1000)
test(x,label)
println("Final insane passed")

include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMax.jl")

println("Profiling SoftMax")
X = rand(100,10)
Y = rand(100,10)

println("First time (compiling...)")
l = SoftMax(Dict{String, Any}("batch_size" => 0, "input_size" => 10))
@time forward(l,X)
@time backward(l,Y)

println("Second time (after compilation...)")
@time forward(l,X)
@time backward(l,Y)

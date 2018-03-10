include("../src/layers/LayerBase.jl")
include("../src/layers/InputLayer.jl")
include("../src/layers/CrossEntropyLoss.jl")

println("Profiling CrossEntropyLoss")
X = rand(100,10)
Y = rand(100,10)
label = rand(100,10)

println("First time (compiling...)")
l = CrossEntropyLoss(Dict{String, Any}("batch_size" => 0, "input_size" => 10), InputLayer((100,10)))
@time forward(l,X,label)
@time backward(l,Y)

println("Second time (after compilation...)")
@time forward(l,X,label)
@time backward(l,Y)

include("../src/layers/LayerBase.jl")
include("../src/layers/Conv.jl")

println("Profiling Conv")
X = rand(32, 3, 30, 30)
Y = rand(32, 128, 28, 28)

println("First time (compiling...)")
@time l = Conv(Dict{String, Any}("batch_size" => 32, "input_size" => (3,30,30)), 128,(3,3))
@time forward(l,X)
@time backward(l,Y)
@time getGradient(l)

println("Second time (after compilation...)")
@time forward(l,X)
@time backward(l,Y)
@time getGradient(l)

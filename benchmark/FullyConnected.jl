include("../src/layers/LayerBase.jl")
include("../src/layers/FullyConnected.jl")

X = rand(500, 784)  #input size 784, batch size 500
Y = rand(500, 800)

println("First time (compiling...)")
@time l = FullyConnected(Dict{String, Any}("batch_size" => 500, "input_size" => [784]), 800)
@time forward(l,X)
@time backward(l,Y)

println("Second time (profiling...)")
@time begin
  for i = 1:10
    forward(l,X)
  end
end
@time begin
  for i = 1:1000
    forward(l,X)
  end
end

@time begin
  for i = 1:10
    backward(l,Y)
  end
end
@time begin
  for i = 1:1000
    backward(l,Y)
  end
end

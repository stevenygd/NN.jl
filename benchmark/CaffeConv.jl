include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")

bsize= 500
X = rand(27, 27, 3,  bsize)
Y = rand(25, 25, 32, bsize)

println("First time (compiling...)")
@time l = CaffeConv(Dict{String, Any}("batch_size" => bsize, "input_size" => (27, 27, 3)), 32, (3,3))
@time y1 = forward(l,X)
@time y1 = backward(l,Y)
@time y1 = getGradient(l)

println("Second time (after compilation) CaffeConv")
X = rand(27, 27, 3,  bsize)
Y = rand(25, 25, 32, bsize)
@time begin
    forward(l,X)
end
@time begin
    backward(l,Y)
end
@time begin
    getGradient(l)
end

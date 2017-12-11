include("../src/NN.jl")
using NN

batch_size = 2
layerX = InputLayer((batch_size,2))
layerY = InputLayer((batch_size, 2))
l1    = DenseLayer(layerX, 2; init_type = "Uniform")
l2    = SoftMaxCrossEntropyLoss(l1, layerY)

l1.W = zeros(l1.W)
println(l1.W)
println(l1.x)
Threads.@threads for i = 1:8
    l = DenseLayer(l1)
    broadcast!(+, l.W, l.W, [1])
    l.x = ones(l.x)
    ccall(:jl_,Void,(Any,), "this is thread number $(Threads.threadid())")
end
println(l1.W)
println(l1.x)

# l3 = DenseLayer(l1)
# println(size(l1.W))

# println(l1.W[1,:])
# println(l3.W[1,:])
# l1.W[1,:] = rand(10, 1)
# println(l1.W[1,:])
# println(l3.W[1,:])
#
# # println(l1.x[1,:])
# # println(l3.x[1,:])
# l1.x = ones(20, 20)
# println(l1.x[1,:])
# println(l3.x[1,:])

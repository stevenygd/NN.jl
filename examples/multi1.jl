include("../src/NN.jl")
using NN

batch_size = 20
layerX = InputLayer((batch_size,20))
layerY = InputLayer((batch_size, 10))
l1    = DenseLayer(layerX, 10; init_type = "Normal")
l2    = SoftMaxCrossEntropyLoss(l1, layerY)

l3 = DenseLayer(l1)
println(size(l1.W))

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

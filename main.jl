include("layers/CrossEntropy.jl")
include("layers/FCLayer.jl")
include("layers/DropoutLayer.jl")
include("layers/ReLu.jl")
include("layers/SequnetialNet.jl")
include("layers/train.jl")

layers = [
    DropoutLayer(0.2, 784),
    FCLayer(784, 800),
    ReLu(),
    DropoutLayer(0.5, 800),
    FCLayer(800, 800),
    ReLu(),
    DropoutLayer(0.5, 800),
    FCLayer(800, 10)
]
criteria = CrossEntropyLoss()
net = SequentialNet(layers, criteria)

using MNIST
features = trainfeatures(1)
label = trainlabel(1)

trainX, trainY = traindata()
testX, testY = testdata()
trX = trainX'
ttl = 500
trX, trY = trX[1:ttl,:], trainY[1:ttl,:]

@assert size(trX)[1] == size(trY)[1]
println(size(trX), size(trY))

# Normalize the input
trX = trX .- repeat(mean(trX, 1), outer = [ttl, 1])

# force to compile the function
train(net, trX, trY, ttl_epo = 1; batch_size = 500)

print("Finish")

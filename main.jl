include("layers/CrossEntropy.jl")
include("layers/FCLayer.jl")
include("layers/ReLu.jl")
include("layers/SequnetialNet.jl")

layers = [
    FCLayer(784, 196),
    ReLu(),
    FCLayer(196, 49),
    ReLu(),
    FCLayer(49, 10)
]
criteria = CrossEntropyLoss()
net = SequentialNet(layers, criteria)

using MNIST
features = trainfeatures(1)
label = trainlabel(1)

trainX, trainY = traindata()
testX, testY = testdata()

trainX = trainX'
ttl = 64
trainX, trainY = trainX[1:ttl,:], trainY[1:ttl,:]

print("Finish")

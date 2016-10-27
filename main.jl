include("layers/CrossEntropy.jl")
include("layers/FCLayer.jl")
include("layers/ReLu.jl")
include("layers/SequnetialNet.jl")
include("layers/train.jl")

layers = [
    # FCLayer(784, 196),
    # ReLu(),
    FCLayer(20, 10),
    ReLu()
]
criteria = CrossEntropyLoss()
net = SequentialNet(layers, criteria)

using MNIST
features = trainfeatures(1)
label = trainlabel(1)

trainX, trainY = traindata()
testX, testY = testdata()

trainX = trainX'
using MultivariateStats

# suppose Xtr and Xte are training and testing data matrix,
# with each observation in a column

# train a PCA model
M = fit(PCA, trainX; maxoutdim=20)

# apply PCA model to testing set
trainX = transform(M, trainX)

ttl = 200
trainX, trainY = trainX[1:ttl,:], trainY[1:ttl,:]

@assert size(trainX)[1] == size(trainY)[1]
println(size(trainX), size(trainY))

# Normalize the input
trainX = trainX - repeat(mean(trainX, 1), outer = [ttl, 1])

train(net, trainX, trainY)

print("Finish")

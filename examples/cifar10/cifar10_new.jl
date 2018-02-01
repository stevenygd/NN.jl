include("../../src/NN.jl")
include("../../util/datasets.jl")
using NN
using Plots
using MLDatasets
trainX, trainY = CIFAR10.traindata()
testX, testY = CIFAR10.testdata()

# Transform Image Vectors from (0, 1) to (-1, 1)
transform(x) = (x .- 0.5) / 0.5
trainX, trainY, testX, testY =
      transform(trainX), transform(trainY), transform(testX), transform(testY)
# Building NN
function Net()
      layerX = InputLayer((32, 32, 3))
      layerY = InputLayer((32, 32, 3))
      l1     = Conv(layerX, 6, (5,5))
      l2     = ReLu(l1)
end

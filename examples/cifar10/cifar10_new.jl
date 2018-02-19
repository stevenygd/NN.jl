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
function Net(batch_size)
      layerX = InputLayer((32, 32, 3, batch_size))
      layerY = InputLayer((32, 32, 3, batch_size))
      c1     = CaffeConv(layerX, 6, (5,5);padding=0, stride=1)
      r1     = ReLu(l1)
      p1     = MaxPool(r1, (2, 2))
      c2     = CaffeConv(p1, 16, (5, 5);padding=0, stride=1)
      r2     = ReLu(c2)
      p2     = MaxPool(r2, (2, 2))
      fl     = Flatten(p2)
      f1     = FullyConnected(fl, 120)
      f2     = FullyConnected(f1, 84)
      f3     = FullyConnected(f2, 10)
      loss   = SoftMaxCrossEntropyLoss(f3, layerY)
      g      = Graph(loss)
      g
end

model = Net(100)

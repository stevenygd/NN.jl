include("../../src/NN.jl")
include("../../util/datasets.jl")
using NN
using Plots
using MLDatasets

# Building NN
function build_model(batch_size)
      layerX = InputLayer((32, 32, 3, batch_size))
      layerY = InputLayer((32, 32, 3, batch_size))
      c1     = CaffeConv(layerX, 6, (5,5);padding=0, stride=1)
      r1     = ReLu(c1)
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
      layerX, layerY, g
end

function get_corr(pred, answ)
    return length(filter(e -> abs(e) < 1e-5, pred-answ))
end


function train(graph::Graph, layerX::Layer, layerY::Layer, optimizer::SgdOptimizer,
      trainX, trainY, batch_size::Int64 = 100, ttl_epo::Int64 = 10)

      N = size(trainX)[4]
      all_losses = []

      for epo=1:ttl_epo
            batch_iter_num = Int(ceil(N/batch_size))
            println("Epo $(epo) num batches : $(batch_iter_num)")
            for batch_iter=1:batch_iter_num
                  local start = (batch_iter-1) * batch_size+1
                  local last = min(N, batch_iter * batch_size)
                  batch_X = trainX[:, :, :, start:last]
                  batch_Y = trainY[start:last, :]
                  loss, pred = optimize(optimizer, Dict(layerX=>batch_X, layerY=>batch_Y))
                  loss = mean(loss)
                  push!(all_losses, loss)
                  batch_cor = get_corr(pred, batch_Y)
                  accu = batch_cor / (last-start+1)
                  println("Epo # $epo, Batch # $(batch_iter), accuracy: $accu, loss: $loss")
            end
      end
      return all_losses
end

function convert_one_hot(label::Array{Int64, 1})
      N = size(label)[1]
      onehot = zeros(size(label)[1], 10)
      for i=1:N
            onehot[i, label[i]+1] = 1.0
      end
      onehot
end

trainX, trainY = CIFAR10.traindata()
testX, testY = CIFAR10.testdata()

# Transform Image Vectors from (0, 1) to (-1, 1)
transform(x) = (x .- 0.5) / 0.5
trainX, testX, = transform(trainX), transform(testX)
# trainY, testY = convert_one_hot(trainY), convert_one_hot(testY)
trainY = convert_one_hot(trainY)

#build model
layerX, layerY, model = build_model(100)

# function lr_schedule(x)
#       lr = 0.0
#       if x < 100
#             lr = 0.002
#       else
#             lr = 0.001
#       end
#       lr
# end

opt = SgdOptimizer(model;base_lr=(x->0.001))

#train and inspection
losses = train(model, layerX, layerY, opt, trainX, trainY, 100, 2)

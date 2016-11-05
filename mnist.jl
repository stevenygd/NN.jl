include("layers/CrossEntropy.jl")
include("layers/SquareLossLayer.jl")
include("layers/FCLayer.jl")
include("layers/DropoutLayer.jl")
include("layers/ReLu.jl")
include("layers/Tanh.jl")
include("layers/SequnetialNet.jl")

using PyPlot
using MNIST

function mnistData()
    features = trainfeatures(1)
    label = trainlabel(1)

    trainX, trainY = traindata()
    N = size(trainX)[2]
    idx = randperm(N)
    trainX = trainX[:, idx]'
    trainY = trainY[idx]

    testX, testY = testdata()
    N = size(testX)[1]
    idx = randperm(N)
    testX = testX[:, idx]'
    testY = testY[idx]

    ttl = 50000
    trX, trY = trainX[1:ttl,:], trainY[1:ttl,:]

    @assert size(trX)[1] == size(trY)[1]
    println(size(trX), size(trY))

    # Normalize the input
    trX = trX .- repeat(mean(trX, 1), outer = [ttl, 1])
    return trX, trY
end

function build_mlp()
    layers = [
        DropoutLayer(0.2),
        FCLayer(784, 800),
        ReLu(),
        DropoutLayer(0.5),
        FCLayer(800, 800),
        ReLu(),
        DropoutLayer(0.5),
        FCLayer(800, 10)
    ]
    criteria = CrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

trX, trY = mnistData()
net = build_mlp()


function train(net::SequentialNet, X, Y; batch_size::Int64 = 64, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    for epo = 1:ttl_epo
        local num_batch = ceil(N/batch_size)
        println("Epo $(epo) num batches : $(num_batch)")
        all_losses = []
        epo_cor = 0
        for bid = 0:(num_batch-1)
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[sidx:eidx,:]
            local batch_Y = Y[sidx:eidx,:]
            loss, pred = forward(net, batch_X, batch_Y)
            loss = mean(loss)
            epo_cor  += length(filter(e -> e == 0, pred - batch_Y))
            local acc = length(filter(e -> e == 0, pred - batch_Y)) / batch_size
            backward(net, batch_Y)
            for i = 1:length(net.layers)
                local layer = net.layers[i]
                local gradi = lrSchedule(epo) * gradient(layer) / batch_size
                local veloc = getLDiff(layer)
                # p - mom*v + (mom*v-lr*g) + mom*(mom*v-lr*g)
                local theta = getParam(layer) - veloc * alpha - gradi + alpha * (alpha * veloc - gradi)
                # local theta = getParam(layer) - gradi + alpha * veloc
                if verbose > 1
                    print("Layer $(i)")
                    print("\tGradient: $(sum(abs(theta - getLDiff(layer))))")
                    if verbose > 1
                        print("\tLastloss: $(sum(abs(layer.last_loss)))")
                    end
                    println()
                end
                setParam!(layer, theta)
            end
            append!(all_losses, loss)
            if verbose > 0
                println("[$(bid)/$(num_batch)]Loss is: $(loss)\tAccuracy:$(acc)")
            end
        end
        local epo_loss = mean(all_losses)
        local epo_accu = epo_cor / N
        append!(epo_losses, epo_loss)
        println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(epo_accu)")
    end
    return forward(net, X, Y)
end

losses, all_losses = train(net, trX, trY, ttl_epo = 100; batch_size = 500,
               lrSchedule = x -> (x < 30) ? 0.03 : 0.01, verbose=0)
plot(1:length(losses), losses)
title("Epoch Losses")
show()


# plot(1:length(all_losses), all_losses)
# title("Batch Losses")
# show()

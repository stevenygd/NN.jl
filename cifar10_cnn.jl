include("src/NN.jl")
include("util/datasets.jl")

using NN
using PyPlot
using IProfile
using MLDatasets

# Multi threaded
# Blas numthreads
batch_size = 500
function build_cnn()
    layers = [
        InputLayer((64,3,32,32)), # dummy batch size
        ConvLayer(32,(5,5)),
        ReLu(),
        ConvLayer(32,(5,5)),
        ReLu(),
        MaxPoolingLayer((2,2)),

        ConvLayer(64,(3,3)),
        ReLu(),
        ConvLayer(64,(3,3)),
        ReLu(),
        MaxPoolingLayer((2,2)),

        FlattenLayer(),
        DropoutLayer(0.5),
        DenseLayer(256),
        ReLu(),
        DropoutLayer(0.5),
        DenseLayer(10)
    ]
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

function get_corr(pred, answ)
    return length(filter(e -> abs(e) < 1e-5, pred-answ))
end

function train(net::SequentialNet, train_set, validation_set;
    batch_size::Int64 = 100, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    local epo_accus = []

    local val_losses = []
    local val_accu   = []
    for epo = 1:ttl_epo
        local num_batch = ceil(N/batch_size)
        if verbose > 0
            println("Epo $(epo) num batches : $(num_batch)")
        end
        all_losses = []
        epo_cor = 0
        for bid = 0:(num_batch-1)
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[sidx:eidx,:,:,:]
            local batch_Y = Y[sidx:eidx,:]
            loss, _ = forward(net, batch_X, batch_Y)
            backward(net, batch_Y)
            append!(all_losses, mean(loss))
            for i = 1:length(net.layers)
                local layer = net.layers[i]
                local gradi = lrSchedule(epo) * getGradient(layer) / batch_size
                local veloc = getVelocity(layer) * alpha - gradi
                local theta = getParam(layer) + alpha * veloc - gradi
                if verbose > 2
                    print("Layer $(i)")
                    print("\tGradient: $(sum(abs(theta - getVelocity(layer))))")
                    if verbose > 1
                        print("\tLastloss: $(sum(abs(layer.dldx))) $(sum(abs(layer.dldy)))")
                    end
                    println()
                end
                setParam!(layer, theta)
            end

            _, pred = forward(net, batch_X, batch_Y; deterministics = true)
            epo_cor  += get_corr(pred, batch_Y)
            local acc = get_corr(pred, batch_Y) / batch_size

            println("[$(bid)/$(num_batch)]Loss is: $(mean(loss))\tAccuracy:$(acc)")
        end
        epo_loss = mean(all_losses)
        epo_accu = epo_cor / N
        append!(epo_losses, epo_loss)
        append!(epo_accus, epo_accu)

        # Run validation set
        val_corr, val_loss, val_ttl = 0, 0., 0
        for t=1:batch_size:length(valX)
            valX_batch, valY_batch = valX[t:t+batch_size-1,:,:,:], valY[t:t+batch_size-1,:]
            @assert dim(valY) == 1
            v_ls, v_pd = forward(net, valX_batch, valY_batch)
            val_loss += mean(v_ls)
            val_corr += get_corr(v_pd, valY_batch)
            val_ttl  += batch_size
        end
        # v_accu = get_corr(v_pd, valY) / v_size
        v_accu = val_corr / val_ttl
        v_loss = v_loss   / val_ttl
        append!(val_losses, v_loss)
        append!(val_accu,   v_accu)

        if verbose > 0
            println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(epo_accu)")
        end
    end
    return epo_losses,epo_accus, val_losses, val_accu
end

trainX, trY = CIFAR10.traindata()
testX,  teY  = CIFAR10.testdata()
println(size(trainX), size(trY), size(testX), size(teY))

# Reshape the fit our convolution layers
trX_all, teX_all = Array{Float64}(50000,3,32,32), Array{Float64}(10000,3,32,32)
permutedims!(trX_all, trainX, [4,3,1,2])
permutedims!(teX_all, testX, [4,3,1,2])
println(size(trX_all), size(teX_all))
trX, valX = trX_all[1:40000,:,:,:], trX_all[40001:end,:,:,:]
trY, valY = trY[1:40000], trY[40001:end]

sigmoid_net = build_cnn()
epo_losses, epo_accu, val_losses, val_accu = train(sigmoid_net, (trX, trY), (valX, valY);
                ttl_epo = 10, batch_size = 64, lrSchedule = x -> 0.01, verbose=1, alpha=0.9)
figure(figsize=(12,6))
subplot(221)
plot(1:length(epo_losses), epo_losses)
title("Training losses (epoch)")

subplot(223)
plot(1:length(epo_accu), epo_accu)
ylim([0, 1])
title("Training Accuracy (epoch)")

subplot(222)
plot(1:length(val_losses), val_losses)
title("Validaiton Losses (epoch)")

subplot(224)
plot(1:length(val_accu), val_accu)
ylim([0, 1])
title("Validaiton Accuracy (epoch)")

show()

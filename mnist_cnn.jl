include("src/NN.jl")
include("util/datasets.jl")
BLAS.set_num_threads(8)

using NN
using PyPlot
using IProfile

# Multi threaded
# Blas numthreads
batch_size = 500
function build_cnn()
    layers = Layer[
        InputLayer((28,28,1,500)),
        # ConvLayer(3,(5,5)),
        # FlatConvLayer(3,(5,5)),
        # MultiThreadedConvLayer(3,(5,5)),
        CaffeConvLayer(3,(5,5)),
        # ReLu(),
        # MaxPoolingLayer((2,2)),

        # ConvLayer(32,(5,5)),
        # FlatConvLayer(32,(5,5)),
        # MultiThreadedConvLayer(32,(5,5)),
        # CaffeConvLayer(32,(5,5)),
        # ReLu(),
        # MaxPoolingLayer((2,2)),

        FlattenLayer(),

        # DropoutLayer(0.5),
        # DenseLayer(256),
        # ReLu(),

        # DropoutLayer(0.5),
        DenseLayer(10)
    ]
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

# function build_cnn_multi_threaded()
#     layers = Layer[
#         InputLayer((1,1,28,28)),
#         ConvLayer(32,(5,5)),
#         ReLu(),
#         MaxPoolingLayer((2,2)),
#         # ConvLayer(32,(5,5)),
#         # ReLu(),
#         # MaxPoolingLayer((2,2)),
#         FlattenLayer(),
#         DropoutLayer(0.5),
#         DenseLayer(256),
#         ReLu(),
#         DropoutLayer(0.5),
#         DenseLayer(10)
#     ]
#     criteria = SoftMaxCrossEntropyLoss()
#     net = SequentialNet(layers, criteria)
#     return net
# end

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
            time_used = @elapsed begin
                batch += 1
                local sidx::Int = convert(Int64, bid*batch_size+1)
                local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
                local batch_X = X[:,:,:,sidx:eidx]
                local batch_Y = Y[sidx:eidx,:]
                loss, _ = forward(net, batch_X, batch_Y)
                backward(net, batch_Y)
                append!(all_losses, mean(loss))
                for i = 1:length(net.layers)
                    layer = net.layers[i]

                    gradi = getGradient(layer)
                    for j = 1:length(gradi)
                        gradi[j] = lrSchedule(epo) * gradi[j] / batch_size
                    end

                    # veloc = getVelocity(layer)
                    # for j = 1:length(veloc)
                    #     veloc[j] = veloc[j] * alpha - gradi[j]
                    # end

                    param = getParam(layer)
                    for j = 1:length(param)
                        # param[j] = param[j] + alpha * veloc[j] - gradi[j]
                        param[j] = param[j] - gradi[j]
                    end
                    if verbose > 2
                        print("Layer $(i)")
                        print("\tGradient: $(sum(abs(theta - getVelocity(layer))))")
                        if verbose > 1
                            print("\tLastloss: $(sum(abs(layer.dldx))) $(sum(abs(layer.dldy)))")
                        end
                        println()
                    end
                    setParam!(layer, param)
                end

                _, pred = forward(net, batch_X, batch_Y; deterministics = true)
                epo_cor  += get_corr(pred, batch_Y)
                local acc = get_corr(pred, batch_Y) / batch_size
            end
            # println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
        end
        local epo_loss = mean(all_losses)
        local epo_accu = epo_cor / N
        append!(epo_losses, epo_loss)
        append!(epo_accus, epo_accu)

        # Run validation set
        v_ls, v_pd = forward(net, valX, valY)
        local v_loss = mean(v_ls)
        v_size = size(valX)[1]
        v_accu = get_corr(v_pd, valY) / v_size
        append!(val_losses, v_loss)
        append!(val_accu,   v_accu)

        if verbose > 0
            println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(epo_accu)")
        end
    end
    return epo_losses,epo_accus, val_losses, val_accu
end

X,Y = mnistData(ttl=55000)
Y = round(Int, Y)
train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]

trX  = permutedims(reshape(trX,  (size(trX,1),  1, 28, 28)), [3,4,2,1])
valX = permutedims(reshape(valX, (size(valX,1), 1, 28, 28)), [3,4,2,1])
teX  = permutedims(reshape(teX,  (size(teX,1),  1, 28, 28)), [3,4,2,1])

net = build_cnn()
# net = build_cnn_multi_threaded()
epo_losses, epo_accu, val_losses, val_accu = train(net, (trX, trY), (valX, valY);
                ttl_epo = 100, batch_size = 500, lrSchedule = x -> 0.01, verbose=1, alpha=0.9)
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

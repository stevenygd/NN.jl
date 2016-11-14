include("layers/SoftMaxCrossEntropy.jl")
include("layers/SquareLossLayer.jl")
include("layers/FCLayer.jl")
include("layers/DropoutLayer.jl")
include("layers/ReLu.jl")
include("layers/Tanh.jl")
include("layers/SequnetialNet.jl")
include("util/datasets.jl")

using PyPlot
function build_mlp_nodrop()
    layers = [
        FCLayer(784, 800),
        ReLu(),
        FCLayer(800, 800),
        ReLu(),
        FCLayer(800, 10)
    ]
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
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
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

function train(net::SequentialNet, X, Y;
    batch_size::Int64 = 64, ttl_epo::Int64 = 10,
    lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
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
            loss, _ = forward(net, batch_X, batch_Y)
            backward(net, batch_Y)
            append!(all_losses, mean(loss))
            for i = 1:length(net.layers)
                local layer = net.layers[i]
                local gradi = lrSchedule(epo) * gradient(layer) / batch_size
                local veloc = getLDiff(layer) * alpha - gradi
                local theta = getParam(layer) + alpha * veloc - gradi
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

            _, pred = forward(net, batch_X, batch_Y; deterministics = true)
            epo_cor  += length(filter(e -> abs(e) < 1e-5, pred - batch_Y))
            local acc = length(filter(e -> abs(e) < 1e-5, pred - batch_Y)) / batch_size

            if verbose > 0
                println("[$(bid)/$(num_batch)]Loss is: $(mean(loss))\tAccuracy:$(acc)")
            end
        end
        local epo_loss = mean(all_losses)
        local epo_accu = epo_cor / N
        append!(epo_losses, epo_loss)
        println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(epo_accu)")
    end
    return epo_losses
end

trX, trY = mnistData(ttl=50000)
net = build_mlp()

losses = train(net, trX, trY, ttl_epo = 100; batch_size = 500,
               lrSchedule = x -> 0.01, verbose=0, alpha=0.9)
plot(1:length(losses), losses)
title("Epoch Losses")
show()


# plot(1:length(all_losses), all_losses)
# title("Batch Losses")
# show()

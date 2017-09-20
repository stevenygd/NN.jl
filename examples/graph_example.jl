include("src/NN.jl")
include("util/datasets.jl")

batch_size = 500

function build_graph()
    l = InputLayer(l, (28,28,1,batch_size)),

    l = CaffeConvLayer(l, 32,(5,5)),
    l = ReLu(),
    l = MaxPoolingLayer(l, (2,2)),

    l = CaffeConvLayer(l, 32,(5,5)),
    l = ReLu(l),
    l = MaxPoolingLayer(l, (2,2)),

    l = FlattenLayer(l),

    l = DenseLayer(l, 256),
    l = ReLu(l),

    l = DropoutLayer(l, 0.5),
    l = DenseLayer(l, 10)

    l = SoftMaxCrossEntropyLoss(l)

    graph = Graph(l)
    return graph
end

function sgd(graph::Graph, train_set, validation_set;
    batch_size::Int64 = 100, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    local epo_accus = []
    all_losses = []

    local val_losses = []
    local val_accu   = []
    for epo = 1:ttl_epo
        local num_batch = ceil(N/batch_size)
        if verbose > 0
            println("Epo $(epo) num batches : $(num_batch)")
        end
        epo_cor = 0
        for bid = 0:(num_batch-1)
            time_used = @elapsed begin
                batch += 1
                local sidx::Int = convert(Int64, bid*batch_size+1)
                local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
                local batch_X = X[:,:,:,sidx:eidx]
                local batch_Y = Y[sidx:eidx,:]
                loss, pred = forward(net, batch_X, batch_Y)
                backward(net, batch_Y)
                append!(all_losses, mean(loss))
                for i = 1:length(net.layers)
                    layer = net.layers[i]

                    param = getParam(layer)
                    if param == nothing
                        continue
                    end

                    gradi = getGradient(layer)
                    for j = 1:length(gradi)
                        param[j] -= lrSchedule(epo) * gradi[j] / batch_size
                    end
                    setParam!(layer, param)
                end

                # _, pred = forward(net, batch_X, batch_Y; deterministics = true)
                epo_cor  += get_corr(pred, batch_Y)
                local acc = get_corr(pred, batch_Y) / batch_size
            end
            println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
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

        # if verbose > 0
            println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(epo_accu)")
        # end
    end
    return epo_losses,epo_accus, val_losses, val_accu, all_losses
end

include("src/NN.jl")
include("util/datasets.jl")
# BLAS.set_num_threads(20)

using NN
using PyPlot
using IProfile

# Multi threaded
# Blas numthreads
batch_size = 500

function build_cnn()
    layers = Layer[
        InputLayer((28,28,1,batch_size)),
        CaffeConvLayer(32,(5,5)),
        ReLu(),
        MaxPoolingLayer((2,2)),

        CaffeConvLayer(32,(5,5)),
        ReLu(),
        MaxPoolingLayer((2,2)),

        FlattenLayer(),

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

function sgd(net::SequentialNet, train_set, validation_set;
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

function nestorove_momemtum(net::SequentialNet, train_set, validation_set;
    batch_size::Int64 = 16, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    all_losses = []
    local epo_losses = []
    local epo_accus = []

    local val_losses = []
    local val_accu   = []
    for epo = 1:ttl_epo
        epo_time_used = @elapsed begin
            local num_batch = ceil(N/batch_size)
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
                    for i = 1:length(net.layers)
                        layer = net.layers[i]
                        param = getParam(layer)
                        if param == nothing
                            continue
                        end
                        gradi = getGradient(layer)
                        veloc = getVelocity(layer)

                        for j = 1:length(param)
                            gradi[j] = lrSchedule(epo) * gradi[j] / batch_size
                            veloc[j] = veloc[j] * alpha - gradi[j]
                            param[j] = param[j] + alpha * veloc[j] - gradi[j]
                        end
                        setParam!(layer, param)
                    end

                    append!(all_losses, mean(loss))
                    epo_cor  += get_corr(pred, batch_Y)
                    local acc = get_corr(pred, batch_Y) / batch_size
                end
                println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
            end

            v_size = size(valX)[1]
            v_loss, v_accu = [],[]
            for i = 1:batch_size:v_size
                batch_X = valX[:,:,:,i:i+batch_size-1]
                batch_Y = valY[i:i+batch_size-1,:]
                curr_v_loss, curr_v_pred = forward(net, batch_X, batch_Y)
                curr_v_accu = get_corr(curr_v_pred, batch_Y) / batch_size
                append!(v_loss, curr_v_loss)
                append!(v_accu, curr_v_accu)
            end
            append!(val_losses, mean(v_loss))
            append!(val_accu,   mean(v_accu))
        end
        println("Epo $(epo) [$(epo_time_used)s] has loss :$(mean(v_loss))\t\taccuracy : $(mean(v_accu))")
    end
    return epo_losses,epo_accus, val_losses, val_accu, all_losses
end

function adagrad(net::SequentialNet, train_set, validation_set;
    batch_size::Int64 = 16, ttl_epo::Int64 = 10, lrSchedule = (x -> (x<7)?0.01:0.001), alpha::Float64 = 0.9, verbose=0)
    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    all_losses = []
    local epo_losses = []
    local epo_accus = []

    local val_losses = []
    local val_accu   = []

    cache = []
    for i = 1:length(net.layers)
        layer = net.layers[i]
        param = getParam(layer)
        if param == nothing
            push!(cache, nothing)
        else
            c = []
            for j = 1:length(param)
                push!(c, zeros(size(param[j])))
            end
            push!(cache, c)
        end
    end;

    for epo = 1:ttl_epo
        epo_time_used = @elapsed begin
            local num_batch = ceil(N/batch_size)
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

                    for i = 1:length(net.layers)
                        layer = net.layers[i]
                        param = getParam(layer)
                        if param == nothing
                            continue # not a learnable layer
                        end

                        grad  = getGradient(layer)
                        for j = 1:length(param)
                            c = cache[i][j]
                            p = param[j]
                            g = grad[j]
                            @assert size(c) == size(p) && size(c) == size(g)
                            c = c + g.^2
                            p = p - lrSchedule(epo) * g ./ (sqrt(c) + 1e-10)
                            cache[i][j] = c
                            param[j] =    p
                        end
                        setParam!(layer, param)
                    end

                    append!(all_losses, mean(loss))
                    epo_cor  += get_corr(pred, batch_Y)
                    local acc = get_corr(pred, batch_Y) / batch_size
                end
                println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
            end

            v_size = size(valX)[1]
            v_loss, v_accu = [],[]
            for i = 1:batch_size:v_size
                batch_X = valX[:,:,:,i:i+batch_size-1]
                batch_Y = valY[i:i+batch_size-1,:]
                curr_v_loss, curr_v_pred = forward(net, batch_X, batch_Y;deterministics=true)
                curr_v_accu = get_corr(curr_v_pred, batch_Y) / batch_size
                append!(v_loss, curr_v_loss)
                append!(v_accu, curr_v_accu)
            end
            append!(val_losses, mean(v_loss))
            append!(val_accu,   mean(v_accu))
        end
        println("Epo $(epo) [$(epo_time_used)s] has loss :$(mean(v_loss))\t\taccuracy : $(mean(v_accu))")
    end
    return epo_losses,epo_accus, val_losses, val_accu, all_losses
end

function RMSprop(net::SequentialNet, train_set, validation_set;
    batch_size::Int64 = 16, ttl_epo::Int64 = 10, lrSchedule = 0.01,
    verbose=0, alpha::Float64 = 0.9)

    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    all_losses = []
    local epo_losses = []
    local epo_accus = []

    local val_losses = []
    local val_accu   = []

    optimizer = RMSPropOptimizer(net, alpha=0.9)
    for epo = 1:ttl_epo
        epo_time_used = @elapsed begin
            local num_batch = ceil(N/batch_size)
            epo_cor = 0
            for bid = 0:(num_batch-1)
                time_used = @elapsed begin
                    batch += 1
                    local sidx::Int = convert(Int64, bid*batch_size+1)
                    local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
                    local batch_X = X[:,:,:,sidx:eidx]
                    local batch_Y = Y[sidx:eidx,:]
                    loss, pred = optimize(optimizer, batch_X, batch_Y)

                    append!(all_losses, mean(loss))
                    epo_cor  += get_corr(pred, batch_Y)
                    local acc = get_corr(pred, batch_Y) / batch_size
                end
                println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
            end

            v_size = size(valX)[1]
            v_loss, v_accu = [],[]
            for i = 1:batch_size:v_size
                batch_X = valX[:,:,:,i:i+batch_size-1]
                batch_Y = valY[i:i+batch_size-1,:]
                curr_v_loss, curr_v_pred = forward(net, batch_X, batch_Y;deterministics=true)
                curr_v_accu = get_corr(curr_v_pred, batch_Y) / batch_size
                append!(v_loss, curr_v_loss)
                append!(v_accu, curr_v_accu)
            end
            append!(val_losses, mean(v_loss))
            append!(val_accu,   mean(v_accu))
        end
        println("Epo $(epo) [$(epo_time_used)s] has loss :$(mean(v_loss))\t\taccuracy : $(mean(v_accu))")
    end
    return epo_losses,epo_accus, val_losses, val_accu,all_losses
end

function Adam(net::SequentialNet, train_set, validation_set;
    batch_size::Int64 = 16, ttl_epo::Int64 = 10, lrSchedule = 0.01,
    beta_1::Float64 = 0.9, beta_2::Float64 = 0.999, verbose=0)
    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    local epo_accus = []

    local val_losses = []
    local val_accu   = []

    optimizer  = AdamOptimizer(net)
    all_losses = []
    for epo = 1:ttl_epo
      epo_time_used = @elapsed begin
        local num_batch = ceil(N/batch_size)
        epo_cor = 0
        for bid = 0:(num_batch-1)
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[:,:,:,sidx:eidx]
            local batch_Y = Y[sidx:eidx,:]
            loss, pred = optimize(optimizer, batch_X, batch_Y)
            append!(all_losses, mean(loss))
            epo_cor  += get_corr(pred, batch_Y)
            local acc = get_corr(pred, batch_Y) / batch_size
        end
      # println("[$(bid)/$(num_batch)]($(epo_time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
        v_size = size(valX)[1]
        v_loss, v_accu = [],[]
        for i = 1:batch_size:v_size
            batch_X = valX[:,:,:,i:i+batch_size-1]
            batch_Y = valY[i:i+batch_size-1,:]
            curr_v_loss, curr_v_pred = forward(net, batch_X, batch_Y;deterministics=true)
            curr_v_accu = get_corr(curr_v_pred, batch_Y) / batch_size
            append!(v_loss, curr_v_loss)
            append!(v_accu, curr_v_accu)
        end
        append!(val_losses, mean(v_loss))
        append!(val_accu,   mean(v_accu))
      end
      println("Epo $(epo) [$(epo_time_used)s] has loss :$(mean(v_loss))\t\taccuracy : $(mean(v_accu))")
    end
    return epo_losses,epo_accus, val_losses, val_accu,all_losses
end

X,Y = mnistData(ttl=55000) # 0-1
println("X statistics: $(mean(X)) $(minimum(X)) $(maximum(X))")

Y = round(Int, Y)
train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]

trX  = permutedims(reshape(trX,  (size(trX,1),  1, 28, 28)), [3,4,2,1])
valX = permutedims(reshape(valX, (size(valX,1), 1, 28, 28)), [3,4,2,1])
teX  = permutedims(reshape(teX,  (size(teX,1),  1, 28, 28)), [3,4,2,1])

println("TrainSet: $(size(trX)) $(size(trY))")
println("ValSet  : $(size(valX)) $(size(valY))")
println("TestSet : $(size(teX)) $(size(teY))")

net = build_cnn()
# net = build_cnn_multi_threaded()
# epo_losses, epo_accu, val_losses, val_accu = train(
#     net, (trX, trY), (valX, valY);
#     ttl_epo = 100, batch_size = batch_size,
#     lrSchedule = x -> 0.01, verbose=1, alpha=0.9
# )

# epo_losses, epo_accu, val_losses, val_accu, all_losses = Adam(
#     net, (trX, trY), (valX, valY);
#     ttl_epo = 10, batch_size = batch_size,
#     lrSchedule = x -> 0.001, verbose=1
# )

epo_losses, epo_accu, val_losses, val_accu, all_losses = Adam(
    net, (trX, trY), (valX, valY);
    ttl_epo = 10, batch_size = batch_size,
    lrSchedule = x -> 0.001, verbose=1
)

# epo_losses, epo_accu, val_losses, val_accu, all_losses = adagrad(
#     net, (trX, trY), (valX, valY);
#     ttl_epo = 10, batch_size = batch_size,
#     lrSchedule = x -> 0.001, verbose=1
# )

# epo_losses, epo_accu, val_losses, val_accu, all_losses = nestorove_momemtum(
#     net, (trX, trY), (valX, valY);
#     ttl_epo = 10, batch_size = batch_size,
#     lrSchedule = x -> 0.01, verbose=1
# )

# epo_losses, epo_accu, val_losses, val_accu, all_losses = sgd(
#     net, (trX, trY), (valX, valY);
#     ttl_epo = 10, batch_size = batch_size,
#     lrSchedule = x -> 0.01, verbose=1
# )

# figure(figsize=(12,6))
# plot(1:length(all_losses), all_losses)
# title("Adam : Training losses")

# figure(figsize=(12,6))
# # subplot(221)
# plot(1:length(all_losses), all_losses)
# title("RMSprop : Training losses")

figure(figsize=(12,6))
# subplot(221)
plot(1:length(all_losses), all_losses)
title("Nestorove Momemtum : Training losses")


# subplot(223)
# plot(1:length(epo_accu), epo_accu)
# ylim([0, 1])
# title("Training Accuracy (epoch)")

# subplot(222)
# plot(1:length(val_losses), val_losses)
# title("Validaiton Losses (epoch)")

# subplot(224)
# plot(1:length(val_accu), val_accu)
# ylim([0, 1])
# title("Validaiton Accuracy (epoch)")

show()

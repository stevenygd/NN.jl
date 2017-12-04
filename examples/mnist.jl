include("../src/NN.jl")
include("../util/datasets.jl")
using NN
batch_size = 500

function build_model()
    input = InputLayer((batch_size,784))
    loss  = InputLayer((batch_size, 10))
    l1    = DenseLayer(input, 1000; init_type = "Normal")
    l2    = ReLu(l1)
    l3    = DropoutLayer(l2, 0.5)
    l4    = DenseLayer(l3, 1000; init_type = "Normal" )
    l5    = ReLu(l4)
    l6    = DropoutLayer(l5, 0.5)
    l7    = DenseLayer(l6, 10; init_type = "Normal")
    l8    = SoftMaxCrossEntropyLoss(l7)
    return input, loss, Graph(l8)
end

function get_corr(pred, answ)
    return length(filter(e -> abs(e) < 1e-5, pred-answ))
end

function sgd(graph::Graph, layerX::Layer, layerY::Layer, optimizer::SgdOptimizerGraph,
    train_set, validation_set;
    batch_size::Int64 = 100, ttl_epo::Int64 = 10,
    lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
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
                local batch_X = X[sidx:eidx, :]
                local batch_Y = Y[sidx:eidx, :]
                loss, pred = optimize(optimizer, Dict(layerX=>batch_X, layerY=>batch_Y))
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
        v_ls, v_pd = forward(graph, valX, valY)
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

function convert_to_one_hot(x::Array{Int64}, classes)
  m = zeros(size(x,1), classes)
  for i=1:size(x,1)
    m[i,x[i]+1]=1
  end
  m
end

X,Y = mnistData(ttl=55000) # 0-1
Y = round.(Int, Y)

println("X statistics: $(mean(X)) $(minimum(X)) $(maximum(X))")


train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]
trY = convert_to_one_hot(trY, 10)
valY = convert_to_one_hot(valY, 10)
teY = convert_to_one_hot(teY, 10)

println("TrainSet: $(size(trX)) $(size(trY))")
println("ValSet  : $(size(valX)) $(size(valY))")
println("TestSet : $(size(teX)) $(size(teY))")

layerX, layerY, graph = build_model()
opt = SgdOptimizerGraph(graph)

epo_losses, epo_accu, val_losses, val_accu, all_losses = sgd(
    graph, layerX, layerY, opt, (trX, trY), (valX, valY);
    ttl_epo = 10, batch_size = batch_size,
    lrSchedule = x -> 0.01, verbose=1
)

figure(figsize=(12,6))
# subplot(221)
plot(1:length(all_losses), all_losses)
title("SGD Graph: Training losses")

show()

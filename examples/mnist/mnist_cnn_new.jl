include("../../src/NN.jl")
include("../../util/datasets.jl")

using NN
using PyPlot
using IProfile

batch_size = 500

function build_model()

    input = InputLayer((28,28,1,batch_size))
    label = InputLayer((batch_size,10))
    c1 = CaffeConv(input, 32, (5,5))
    r1 = ReLu(c1);
    m1 = MaxPoolingLayer(r1, (2,2))
    c2 = CaffeConv(m1, 32, (5,5))
    r2 = ReLu(c2);
    m2 = MaxPoolingLayer(r2, (2,2))
    fl = Flatten(m2)
    fc1 = FullyConnected(fl, 256)
    rf = ReLu(fc1);
    d = DropoutLayer(rf, 0.5)
    fc2 = FullyConnected(d, 10)
    loss = SoftMaxCrossEntropyLoss(fc2, label)

    return input, label, Graph(loss)
end

function get_cor(pred, label)
    @assert size(pred) == size(label)
    cor = 0
    for i=1:(size(pred)[1])
        idx = findmax(pred[i, :])[2]
        pred_idx = findmax(label[i, :])[2]
        if idx == pred_idx
            cor += 1
        end
    end
    return cor
end

function sgd(graph::Graph, layerX::Layer, layerY::Layer, optimizer::SgdOptimizer,
    train_set, validation_set;
    batch_size::Int64 = 100, ttl_epo::Int64 = 10, alpha::Float64 = 0.9)

    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    local epo_accus = []
    all_losses = []

    local val_losses = []
    local val_accus   = []

    for epo = 1:ttl_epo
        epo_time = @elapsed begin
            local num_batch = ceil(N/batch_size)
            println("Epo $(epo) num batches : $(num_batch)")

            epo_cor = 0
            for bid = 0:(num_batch-1)
                time_used = @elapsed begin
                    batch += 1
                    local start::Int = bid*batch_size+1
                    local last::Int = min(N, (bid+1)*batch_size)
                    local batch_X = X[:,:,:,start:last]
                    local batch_Y = Y[start:last, :]
                    loss, pred = optimize(optimizer, Dict(layerX=>batch_X, layerY=>batch_Y))
                    push!(all_losses, mean(loss))
                    batch_cor = get_cor(pred, batch_Y)
                    epo_cor  += batch_cor
                    local accu = batch_cor / (last-start+1)
                end
                println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(accu)")
            end
            local epo_loss = mean(all_losses)
            local epo_accu = epo_cor / N
            push!(epo_losses, epo_loss)
            push!(epo_accus, epo_accu)

            # Run validation set
            println("Validation size: $(size(valY))")
            val_loss, val_pred = forward(graph, Dict(layerX=>valX, layerY=>valY))
            val_loss = mean(val_loss)
            val_size = size(valY)[1]
            val_accu = get_cor(val_pred, valY) / val_size
            push!(val_losses, val_loss)
            push!(val_accus, val_accu)
        end
        println("Epo $(epo) takes ($(epo_time)s), validation set has loss :
                $(mean(val_loss))\t\taccuracy : $(val_accu)")
    end

    return epo_losses, epo_accus, val_losses, val_accus, all_losses
end

X,Y = mnistData(ttl=55000) # 0-1
# println("X statistics: $(mean(X)) $(minimum(X)) $(maximum(X))")

train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]
println("TrainSet: $(size(trX)) $(size(trY))")
println("ValSet  : $(size(valX)) $(size(valY))")
println("TestSet : $(size(teX)) $(size(teY))")

trX  = permutedims(reshape(trX,  (size(trX,1),  1, 28, 28)), [3,4,2,1])
valX = permutedims(reshape(valX, (size(valX,1), 1, 28, 28)), [3,4,2,1])
teX  = permutedims(reshape(teX,  (size(teX,1),  1, 28, 28)), [3,4,2,1])

layerX, layerY, graph = build_model()
opt = SgdOptimizer(graph;base_lr=(x->0.001))

epo_losses, epo_accu, val_losses, val_accu, all_losses = sgd(
graph, layerX, layerY, opt, (trX, trY), (valX, valY);
ttl_epo = 10, batch_size = batch_size)

# subplot(221)
println("size: $(size(val_losses))")
println("size: $(size(epo_losses))")
println("size: $(size(epo_accu))")
p=plot([all_losses, val_losses, epo_accu, val_accu],
        xlabel=["batch" "epoch" "epoch" "epoch"],
        ylabel=["losses" "loss" "accuracy" "accuracy"],
        title =["train losess" "validation losses" "train accuracy" "validation accuracy"],
        layout=4)

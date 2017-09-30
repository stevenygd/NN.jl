include("../src/NN.jl")
include("../util/datasets.jl")

using NN
using PyPlot
using IProfile

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
  m = size(pred)[1]
  p = zeros(m)
  l = zeros(m)
  for i = 1:m
      p[i] = findmax(pred[i,:])[2] - 1
      l[i] = findmax(answ[i,:])[2] - 1
  end
  return length(filter(e -> abs(e) < 1e-5, p-l))
end

function training(net::SequentialNet, optimizer, train_set, validation_set;
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
            println("[$(bid)/$(num_batch)] Loss is: $(mean(loss))\tAccuracy:$(acc)")
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

X,Y = mnistData(ttl=55000) # 0-1
# println("X statistics: $(mean(X)) $(minimum(X)) $(maximum(X))")

function convert_to_one_hot(x::Array{Int64}, classes)
  m = zeros(size(x,1), classes)
  for i=1:size(x,1)
    m[i,x[i]+1]=1
  end
  m
end

Y = round(Int, Y)
train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], covertToMatrix(train_set[2],10)
valX, valY = validation_set[1], covertToMatrix(validation_set[2],10)
teX, teY = test_set[1], covertToMatrix(test_set[2],10)

trX  = permutedims(reshape(trX,  (size(trX,1),  1, 28, 28)), [3,4,2,1])
valX = permutedims(reshape(valX, (size(valX,1), 1, 28, 28)), [3,4,2,1])
teX  = permutedims(reshape(teX,  (size(teX,1),  1, 28, 28)), [3,4,2,1])

println("TrainSet: $(size(trX)) $(size(trY))")
println("ValSet  : $(size(valX)) $(size(valY))")
println("TestSet : $(size(teX)) $(size(teY))")

net = build_cnn()
bdam_optimizer  = BdamOptimizer(net)

bdam_epo_losses, bdam_epo_accu, bdam_val_losses, bdam_val_accu, bdam_all_losses = training(
    net, bdam_optimizer, (trX, trY), (valX, valY);
    ttl_epo = 1, batch_size = batch_size,
    lrSchedule = x -> 0.001, verbose=1
)

net = build_cnn()
adam_optimizer  = AdamOptimizer(net)

adam_epo_losses, adam_epo_accu, adam_val_losses, adam_val_accu, adam_all_losses = training(
    net, adam_optimizer, (trX, trY), (valX, valY);
    ttl_epo = 1, batch_size = batch_size,
    lrSchedule = x -> 0.001, verbose=1
)


figure(figsize=(12,6))
plot(1:length(bdam_losses), bdam_losses,  label="Bdam")
plot(1:length(adam_losses), adam_losses, label="ADAM")
ylim([0, 1.5])
xlabel("batches (size=500,total 1 epoches)")
ylabel("loss")
title("Training losses with different optimizers")
legend(loc="upper right",fancybox="true")
savefig("optimizers.png")
show()

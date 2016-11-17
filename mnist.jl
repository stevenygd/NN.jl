using PyPlot
using IProfile

include("layers/SoftMaxCrossEntropy.jl")
include("layers/SquareLossLayer.jl")
include("layers/FCLayer.jl")
include("layers/DropoutLayer.jl")
include("layers/ReLu.jl")
include("layers/Tanh.jl")
include("layers/SequnetialNet.jl")
include("util/datasets.jl")

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
function get_corr(pred, answ)
    return length(filter(e -> abs(e) < 1e-5, pred-answ))
end

function train(net::SequentialNet, train_set, validation_set; batch_size::Int64 = 64, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9, verbose=0)
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
        tftime, tbtime, tgtime = 0., 0., 0.
        tt1, tt2, tt3, tt4 = 0., 0., 0., 0.
        local etime = @elapsed begin
          for bid = 0:(num_batch-1)
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[sidx:eidx,:]
            local batch_Y = Y[sidx:eidx,:]

            local ftime = @elapsed begin
              loss, _ = forward(net, batch_X, batch_Y)
              append!(all_losses, mean(loss))
            end
            tftime += ftime
            local btime = @elapsed begin
              backward(net, batch_Y)
            end
            tbtime += btime
            local gtime = @elapsed begin
              for i = 1:length(net.layers)
                local t1 = @elapsed begin
                  local layer = net.layers[i]
                  local gradi = lrSchedule(epo) * gradient(layer) / batch_size
                end
                tt1 += t1
                local t2 = @elapsed begin
                  local veloc = getLDiff(layer) * alpha - gradi
                end
                tt2 += t2
                local t3 = @elapsed begin
                  local theta = getParam(layer) + alpha * veloc - gradi
                end
                tt3 += t3
                local t4 = @elapsed begin
                  setParam!(layer, theta)
                end
                tt4 += t4
              end
            end
            tgtime += gtime
            # println("Train $(batch_size) sized batch takes $(btime) s.")
          end
        end
        println("Epoch $(epo) training time: $(tftime + tbtime + tgtime) s.")
        println("Decomposition: $(tftime), $(tbtime), $(tgtime)")
        println("Gradient: $(tt1) $(tt2) $(tt3) $(tt4)")

        local ttime = @elapsed begin
          local epo_loss = mean(all_losses)
          # epo_cor = 0
          # for bid = 0:(num_batch-1)
          #   batch += 1
          #   local sidx::Int = convert(Int64, bid*batch_size+1)
          #   local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
          #   bX, bY  = X[sidx:eidx,:], Y[sidx:eidx,:]
          #   _, pred = forward(net, bX, bY)
          #   epo_cor+= get_corr(pred, bY)
          # end
          # local epo_accu = epo_cor / N
          append!(epo_losses, epo_loss)
          # append!(epo_accus, epo_accu)

          # Run validation set
          v_ls, v_pd = forward(net, valX, valY)
          local v_loss = mean(v_ls)
          v_size = size(valX)[1]
          v_accu = get_corr(v_pd, valY) / v_size
          append!(val_losses, v_loss)
          append!(val_accu,   v_accu)
        end
        println("Epoch $(epo) testing time: $(etime) s.")
        println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(v_accu)")
    end
    return epo_losses,epo_accus, val_losses, val_accu
end

function demoTrain(trX, trY, valX, valY, teX, teY;total_epo=100)
    net = build_mlp()
    epo_losses, epo_accu, val_losses, val_accu = train(net, (trX, trY), (valX, valY);
                ttl_epo = total_epo, batch_size = 500, lrSchedule = x -> 0.01, verbose=1, alpha=0.9)

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

    train_loss, pred = forward(net, trX, trY; deterministics = true)
    N = size(trX)[1]
    right_idx = filter(i-> abs(pred[i] - trY[i]) <  1e-5, 1:N)
    wrong_idx = filter(i-> abs(pred[i] - trY[i]) >= 1e-5, 1:N)

end


X,Y = mnistData(ttl=55000)
train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]


net = build_mlp()
train(net, (trX, trY), (valX, valY); ttl_epo = 1, batch_size = 500)
Profile.clear()
Profile.init()
@profile begin train(net, (trX, trY), (valX, valY); ttl_epo = 3, batch_size = 500) end
Profile.print(open("profile_info","w"))
Profile.print()

include("SequnetialNet.jl")

function train(net::SequentialNet, X, Y; batch_size::Int64 = 64, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9)
    local verbose = 0
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    for epo = 1:ttl_epo
        local num_batch = ceil(N/batch_size)
        println("Epo $(epo) num batches : $(num_batch)")
        all_losses = []
        epo_cor = 0
        for bid = 0:num_batch
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
                local gradi = lrSchedule(epo) * gradient(layer)
                local veloc = getLDiff(layer)
                # p - mom*v + (mom*v-lr*g) + mom*(mom*v-lr*g)
                # local theta = getParam(layer) - veloc * alpha - gradi + alpha * (alpha * veloc - gradi)
                local theta = getParam(layer) - gradi + alpha * veloc
                if verbose > 0
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
            if verbose > 1
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

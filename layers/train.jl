function train(net::SequentialNet, X, Y;
    batch_size::Int64 = 64, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9)
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    for epo = 1:ttl_epo
        println("Epo $(epo):")
        local num_batch = ceil(N/batch_size)-1
        println("NUMBER OF BATCH:$(num_batch)")
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
                local delta = lrSchedule(epo) * gradient(layer) / batch_size
                local momen = alpha * getLDiff(layer)
                local theta = getParam(layer) - delta + momen
                setParam!(layer, theta)
                # println("Gradient:$(sum(abs(delta)))")
            end
            append!(all_losses, loss)
            println("[$(bid)/$(num_batch)]Loss is: $(loss)\tAccuracy:$(acc)")
        end
        local epo_loss = mean(all_losses)
        local epo_accu = epo_cor / N
        append!(epo_losses, epo_loss)
        println("Epo $(epo) has loss     : $(epo_loss)")
        println("Epo $(epo) has accuracy : $(epo_accu)")
    end
end

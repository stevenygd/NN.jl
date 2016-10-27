function train(net::SequentialNet, X, Y; batch_size::Int64 = 64, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01), alpha::Float64 = 0.9)
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    for epo = 1:ttl_epo
        println("Epo $(epo):")
        local num_batch = ceil(N/batch_size)-1
        println("NUMBER OF BATCH:$(num_batch)")
        all_losses = []
        for bid = 0:num_batch
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[sidx:eidx,:]
            local batch_Y = Y[sidx:eidx,:]
            local loss = mean(forward(net, batch_X, batch_Y))
            backward(net, batch_Y)
            for i = 1:length(net.layers)
                local layer = net.layers[i]
                local theta = getParam(layer) - lrSchedule(epo) * gradient(layer) / batch_size + alpha * getLDiff(layer)
                setParam!(layer, theta)
            end
            append!(all_losses, loss)
            println("[$(bid)/$(num_batch)]Loss is: $(loss)")
        end
        local epo_loss = mean(all_losses)
        append!(epo_losses, epo_loss)
        println("Epo $(epo) has loss : $(epo_loss)")
    end
end

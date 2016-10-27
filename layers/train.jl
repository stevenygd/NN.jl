function sgd(net::SequentialNet, batch_X, batch_Y; lr::Float64 = 0.0001)
    local batch_size = size(batch_X)[1]
    local ttl_loss   = 0.
    local gradients  = []
    for i = 1:length(net.layers)
        local layer = net.layers[i]
        append!(gradients,zeros(size(getParam(layer))))
    end
    for b = 1:batch_size
        local X = batch_X[b,:] 
        local Y = batch_Y[b,:]
        local loss = forward(net, X, Y) # Propogate the input and output, calculate the loss
        #println(net.layers[1].last_output)
        backward(net, Y) # Propagate the dldy
        for i = 1:length(net.layers)
            gradients[i] += gradient(net.layers[i]) 

        end
        ttl_loss += loss
    end
    # println(gradients)
    for i = 1:length(net.layers)
        local layer = net.layers[i]
        local theta = getParam(layer) - lr * gradients[i] / batch_size
        setParam!(layer, theta)
    end

    return ttl_loss
end

function train(net::SequentialNet, X, Y; ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01))
    local batch_size = 128
    local N = size(Y)[1]
    local batch=0
    for epo = 1:ttl_epo
        println("Epo $(epo):")
        local num_batch = ceil(N/batch_size)-1
        println("NUMBER OF BATCH:$(num_batch)")
        for bid = 0:num_batch
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[sidx:eidx,:]
            local batch_Y = Y[sidx:eidx,:]
            local loss = sgd(net, batch_X, batch_Y; lr=lrSchedule(epo))
            println("[$(bid)/$(num_batch)]Loss is: $(loss)")
        end
    end
end

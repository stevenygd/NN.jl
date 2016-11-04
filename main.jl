include("layers/CrossEntropy.jl")
include("layers/SquareLossLayer.jl")
include("layers/FCLayer.jl")
include("layers/DropoutLayer.jl")
include("layers/ReLu.jl")
include("layers/Tanh.jl")
include("layers/SequnetialNet.jl")
using PyPlot
using MNIST

function mnistData()
    features = trainfeatures(1)
    label = trainlabel(1)

    trainX, trainY = traindata()
    testX, testY = testdata()
    trX = trainX'
    ttl = 40000
    trX, trY = trX[1:ttl,:], trainY[1:ttl,:]

    @assert size(trX)[1] == size(trY)[1]
    println(size(trX), size(trY))

    # Normalize the input
    trX = trX .- repeat(mean(trX, 1), outer = [ttl, 1])
    return trX, trY
end

function selfGenerateData(N1, N2)
    N = N1+ N2

    points = Array{Float64}(N,2)

    ## gaussian center
    center, sigma = [0,0], [1 0; 0 1]
    points[:,1] = randn(N, 1)
    points[:,2] = randn(N, 1)

    ## generate the spherical
    r = 10. 
    for i in (N1+1) : N
        theta = randn(1, 1)[1] * 2 * pi
        for j in 1 : 2
            if(j % 2 == 0)
                points[i,j] = r * sin(theta) + randn(1, 1)[1] * 0.5
            else
                points[i,j] = r * cos(theta) + randn(1, 1)[1] * 0.5
            end
        end
    end

    Y = ones(N, 1)
    Y[1:N1, :] = zeros(N1,1)

    ALL = hcat(points, Y)
    ALL = ALL[randperm(N),:]

    trX = ALL[:, 1:2]
    trY = ALL[:, 3]
    # scatter(points[:,1], points[:,2])

    return trX, trY
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
    criteria = CrossEntropyLoss()
    net = SequentialNet(layers, criteria)
end

function build_simple_net()
    layers = [
        FCLayer(2,4;init_type="Random"),
        ReLu(),
        FCLayer(4,2;init_type="Random"),
        ReLu(),
        FCLayer(2,1;init_type="Random"),
        Tanh()
    ]
    criteria = SquareLossLayer()
    net = SequentialNet(layers, criteria)
end

function train(net::SequentialNet, X, Y; batch_size::Int64 = 64, ttl_epo::Int64 = 10, lrSchedule = (x -> 0.01))
    local verbose = 0
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    local all_losses = []
    for epo = 1:ttl_epo
        local num_batch = ceil(N/batch_size)
        println("Epo $(epo) num batches : $(num_batch)")
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
                local theta = getParam(layer) - gradi
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
        # @printf "Epo %d has loss:\t%.3f\taccurarcy:%.3f" epo epo_loss epo_accu
        println("Epo $(epo) has loss :$(epo_loss)\t\taccuracy : $(epo_accu)")
    end
    return epo_losses, all_losses
end

# trX, trY = mnistData()
trX, trY = selfGenerateData(50,50)
scatter(trX[:,1], trX[:,2], c=map(x-> (x >0)?"r":"g",trY))
title("Data")
show()

net = build_simple_net()

losses, all_losses = train(net, trX, trY, ttl_epo = 5000; batch_size = 100,
               lrSchedule = x -> 0.03)
plot(1:length(losses), losses)
title("Epoch Losses")
show()


# plot(1:length(all_losses), all_losses)
# title("Batch Losses")
# show()

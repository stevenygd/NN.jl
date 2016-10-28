include("layers/CrossEntropy.jl")
include("layers/FCLayer.jl")
include("layers/DropoutLayer.jl")
include("layers/ReLu.jl")
include("layers/SequnetialNet.jl")
include("layers/train.jl")
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

function selfGenerateData()
    N1, N2 = 1000, 1000
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
                points[i,j] = r * sin(theta) + randn(1, 1)[1] * 2
            else
                points[i,j] = r * cos(theta) + randn(1, 1)[1] * 2
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

trX, trY = mnistData()
# trX, trY = selfGenerateData()
loss, pred = train(net, trX, trY, ttl_epo = 10000; batch_size = 500, lrSchedule = (x->0.01), alpha=0.9)
correct = length(filter(x -> x == 0, pred - trY))
println("Accuracy is :$(correct/length(trY))")

println("Finish")


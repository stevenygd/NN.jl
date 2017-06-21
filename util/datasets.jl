using MNIST
include("./CIFAR.jl")
import CIFAR

function cifarData(;ttl= 50000)
    trainX = Array(Float64, 3072,50000)
    trainY = Array(Float64, 50000,1)
    for i in 1 : 5
        idx_start = (i-1)*10000 + 1
        idx_end   =  i * 10000
        trainX[:, idx_start:idx_end], trainY[idx_start:idx_end], labels = CIFAR.traindata(batch_number = i)
    end
    N = size(trainX)[2]
    idx = randperm(N)
    trainX = trainX[:,idx]'
    trainY = trainY[idx]
    X = zeros(N,3,32,32)
#    print(trainX[1,:])
    for i in 1 : N
        X[i,:,:,:] = channelize(trainX[i,:])
    end

    X = X[1:ttl,:,:,:]
    Y = trainY[1:ttl,]
    return X,Y
end

function cifarLables()
    return CIFAR.labelnames()
end

function channelize(sample)
#    @assert size(sample)[1] = 3072
    r = sample[1:1024]
    g = sample[1025:2048]
    b = sample[2049:3072]

    r = reshape(r,32,32)'
    g = reshape(g,32,32)'
    b = reshape(b,32,32)'
    channel = zeros(3,32,32)
    channel[1,:,:] = r
    channel[2,:,:] = g
    channel[3,:,:] = b
    return channel
end

function mnistData(;ttl=55000)
    features = trainfeatures(1)
    label = trainlabel(1)

    trainX, trainY = traindata()
    N = size(trainX)[2]
    idx = randperm(N)
    trainX = trainX[:, idx]'
    trainY = trainY[idx]

    testX, testY = testdata()
    N = size(testX)[1]
    idx = randperm(N)
    testX = testX[:, idx]'
    testY = testY[idx]

    X, Y = trainX[1:ttl,:], trainY[1:ttl,:]

    @assert size(X)[1] == size(Y)[1]
    # println(size(trX), size(trY))

    # Tiny bit of preprocessing, the image will be put in range of 0..1
    # print(X[1,:])
    X = X / 256.
    Y = round(Int,Y)
    return X, Y
end

function sphericalDataSet(N1, N2)
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
    Y[1:N1, :] = -ones(N1,1)

    ALL = hcat(points, Y)
    ALL = ALL[randperm(N),:]

    trX = ALL[:, 1:2]
    trY = ALL[:, 3]
    return trX, trY
end

function XORData(size; dist=0.1, scale = 1.)
    ttl = size * 4
    Y = -ones(ttl, 1)
    Y[1:size*2,:] = ones(size*2,1)

    oneWithDist = scale + dist
    zeroWithDist = 0 + dist
    X = rand(ttl, 2) * scale
    X[1:size,:]            = broadcast(+,X[1:size,:],[zeroWithDist zeroWithDist])
    X[(size+1):2*size,:]   = broadcast(+,X[(size+1):2*size,:],[-oneWithDist -oneWithDist])
    X[(2*size+1):3*size,:] = broadcast(+,X[(2*size+1):3*size,:],[-oneWithDist zeroWithDist])
    X[(3*size+1):4*size,:] = broadcast(+,X[(3*size+1):4*size,:],[zeroWithDist -oneWithDist])

    newIdx = randperm(size*4)
    X = X[newIdx,:]
    Y = Y[newIdx,:]
    return X,Y
end

function datasplit(trX, trY; ratio = 0.8)
    N = size(trX)[1]
    size_training = convert(Int, ceil(N * ratio))
    size_testing  = convert(Int, ceil(N * (1-ratio) * 0.5))
    train_set = (trX[1:size_training,:],trY[1:size_training,:])
    test_set  = (trX[size_training + 1:size_training + size_testing, :],
                 trY[size_training + 1 : size_training + size_testing, :])
    validation_set = (trX[size_training + size_testing + 1:N, :],
                      trY[size_training + size_testing + 1:N,:])
    #TODO might need to ensure label to be matrix
    return train_set, test_set, validation_set
end

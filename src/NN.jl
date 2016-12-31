module NN
    export SequentialNet, forward, backward, getGradient,
           getParam, setParam!, getVelocity
    export Layer, DropoutLayer, DenseLayer, ReLu, Sigmoid, SoftMax,
           SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh, InputLayer,
           ConvLayer, MaxPoolingLayer, FlattenLayer

    include("layers/LayerBase.jl")
    include("layers/InputLayer.jl")
    include("layers/DropoutLayer.jl")
    include("layers/DenseLayer.jl")
    include("layers/ReLu.jl")
    include("layers/Sigmoid.jl")
    include("layers/SoftMax.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/SquareLossLayer.jl")
    include("layers/Tanh.jl")
    include("layers/SequentialNet.jl")
    include("layers/ConvLayer.jl")
    include("layers/MaxPoolingLayer.jl")
    include("layers/FlattenLayer.jl")
end

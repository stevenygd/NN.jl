module NN
    export SequentialNet, forward, backward, gradient,
           getParam, setParam!, getVelocity
    export Layer, DropoutLayer, FCLayer, ReLu, Sigmoid, SoftMax,
           SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh

    include("layers/Base.jl")
    include("layers/DropoutLayer.jl")
    include("layers/FCLayer.jl")
    include("layers/ReLu.jl")
    include("layers/Sigmoid.jl")
    include("layers/SoftMax.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/SquareLossLayer.jl")
    include("layers/Tanh.jl")
    include("layers/SequentialNet.jl")
end

module NN
    # export Layer, DropoutLayer, DenseLayer, ReLu, Sigmoid, SoftMax,
    #        SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh, InputLayer,
    #        Conv, MaxPool, Flatten,
    #        CaffeConv, CrossEntropyLoss
    export Graph, forward, backward, getGradient,
	    getParam, setParam!, getVelocity, getNumParams
    export Layer, LearnableLayer, DataLayer, FullyConnected, FullyConnectedFixed,
            ReLu, DropoutLayer,
            SoftMaxCrossEntropyLoss, InputLayer, MaxPool,
            Flatten, Conv
    export BlockFixedArray, quantize, BlockFixedFC
    include("layers/LayerBase.jl")
    include("layers/InputLayer.jl")
    include("layers/DropoutLayer.jl")
    include("layers/FullyConnected.jl")
    include("layers/FullyConnectedFixed.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/ReLu.jl")
    include("layers/SoftMax.jl")
    include("layers/SquareLossLayer.jl")
    include("layers/MaxPool.jl")
    include("layers/Flatten.jl")
    include("layers/Conv.jl")
    include("layers/Graph.jl")
    # include("layers/CrossEntropyLoss.jl")

    # optimizers
    # include("optimizers/Adam.jl")
    # include("optimizers/AdamPrim.jl")
    # include("optimizers/RMSprop.jl")
    # include("optimizers/SGD.jl")
    include("optimizers/SGD.jl")
    include("fixed_point_number/BlockFixedArray.jl")
    include("layers/BlockFixedFC.jl")
    # export AdamOptimizer, AdamPrimOptimizer, BdamOptimizer, CdamOptimizer, DdamOptimizer,
    #        RMSPropOptimizer, SgdOptimizer, optimize
    export SgdOptimizer, optimize
end

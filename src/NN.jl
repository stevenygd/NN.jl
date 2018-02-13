module NN
    # export SequentialNet, Graph, forward, backward, getGradient,
    #        getParam, setParam!, getVelocity, getNumParams
    # export Layer, DropoutLayer, DenseLayer, ReLu, Sigmoid, SoftMax,
    #        SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh, InputLayer,
    #        Conv, MaxPoolingLayer, Flatten,
    #        CaffeConv, CrossEntropyLoss
    export Graph, forward, backward, getGradient,
	    getParam, setParam!, getVelocity, getNumParams
    export Layer, DataLayer, FullyConnected, ReLu, DropoutLayer,
	    SoftMaxCrossEntropyLoss, InputLayer, MaxPoolingLayer, CaffeConv,
        Flatten
    include("layers/LayerBase.jl")
    include("layers/Graph.jl")
    include("layers/InputLayer.jl")
    include("layers/DropoutLayer.jl")
    include("layers/FullyConnected.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/ReLu.jl")
    include("layers/SoftMax.jl")
    include("layers/SquareLossLayer.jl")
    include("layers/MaxPoolingLayer.jl")
    include("layers/Flatten.jl")
    include("layers/CaffeConv.jl")
    # include("layers/CrossEntropyLoss.jl")

    # optimizers
    # include("optimizers/Adam.jl")
    # include("optimizers/AdamPrim.jl")
    # include("optimizers/RMSprop.jl")
    # include("optimizers/SGD.jl")
    include("optimizers/SGD.jl")
    # export AdamOptimizer, AdamPrimOptimizer, BdamOptimizer, CdamOptimizer, DdamOptimizer,
    #        RMSPropOptimizer, SgdOptimizer, optimize
    export SgdOptimizer, optimize
end

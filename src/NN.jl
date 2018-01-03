module NN
    # export SequentialNet, Graph, forward, backward, getGradient,
    #        getParam, setParam!, getVelocity, getNumParams
    # export Layer, DropoutLayer, DenseLayer, ReLu, Sigmoid, SoftMax,
    #        SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh, InputLayer,
    #        Conv, MaxPoolingLayer, FlattenLayer,
    #        CaffeConv, CrossEntropyLoss
    export Graph, forward, backward, getGradient,
	    getParam, setParam!, getVelocity, getNumParams
    export Layer, DataLayer, DenseLayer, ReLu, DropoutLayer,
	    SoftMaxCrossEntropyLoss, InputLayer
    include("layers/LayerBase.jl")
    include("layers/Graph.jl")
    include("layers/InputLayer.jl")
    include("layers/DropoutLayer.jl")
    include("layers/DenseLayer.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/ReLu.jl")
    # include("layers/Sigmoid.jl")
    # include("layers/SoftMax.jl")
    # include("layers/SquareLossLayer.jl")
    # include("layers/Tanh.jl")
    # include("layers/SequentialNet.jl")
    include("layers/Conv.jl")
    # include("layers/MaxPoolingLayer.jl")
    # include("layers/FlattenLayer.jl")
    # include("layers/FlatConv.jl")
    # include("layers/MultiThreadedConv.jl")
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

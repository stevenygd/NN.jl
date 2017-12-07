include("../NN.jl")
using NN


mainLayer = DenseLayer(Dict("input_size"=>28, "batch_size"=>10), 10; init_type="Normal")

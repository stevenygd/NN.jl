include("../src/layers/ConvLayer.jl")
import Calculus: check_gradient
using Base.Test

# TODO: this example has stride == 1
l = ConvLayer(2,(3,3))
kern = zeros(2,3,3,3)
kern[1,1,:,:] = [0 -1 -1; 0  -1 -1; 0  1  -1.]
kern[1,2,:,:] = [1 0  1 ; -1 1  1 ; 1  1  1. ]
kern[1,3,:,:] = [0 -1 0 ; -1 0  0 ; 0  0  0. ]
kern[2,1,:,:] = [0 -1 0 ; 1  0  0 ; 1  -1 1. ]
kern[2,2,:,:] = [0 0  0 ; 0  1  -1; -1 0  -1.]
kern[2,3,:,:] = [0 0  0 ; 0  1  1 ; 1  -1 1. ]

bias = [1 0.]
X = zeros(Float64,1,3,7,7)
X[1,1,2:6,2:6] = [
    2 2 0 0 0;
    1 1 0 1 2;
    0 2 0 0 2;
    0 0 1 0 2;
    1 0 2 1 2
]
X[1,2,2:6,2:6] = [
    2 0 1 0 1;
    2 1 2 0 1;
    1 1 0 2 0;
    0 0 1 0 2;
    1 2 1 2 1
]
X[1,3,2:6,2:6] = [
    2 2 2 0 2;
    2 1 0 2 1;
    2 2 1 2 1;
    2 2 2 1 0;
    1 2 0 0 2
]
Y = zeros(Float64, 1, 2, 3, 3)
Y[1,1,:,:] = [
    2  2  5  ;
    -2 2  -4 ;
    1  -6 -4
]
Y[1,2,:,:] = [
    4  9  3  ;
    3  3  -2 ;
    2  -2 2
]

init(l, nothing, Dict{String, Any}("batch_size" => 1, "input_size" => (3,7,7)))
y = forward(l,X)
@test size(y) == size(Y)
@test y == Y
# @time backward(l,Y)
# @time getGradient(l)

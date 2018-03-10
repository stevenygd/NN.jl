include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")
using Base.Test

batch_size = 1
input_size = (5,5,3)

l = CaffeConv(Dict("batch_size" => batch_size, "input_size" => input_size), 2, (3,3); padding=1, stride=2)

x = zeros((5,5,3,1))
x[:,:,1,1] = [0 1 0 1 1;1 0 2 2 2;0 0 1 0 0;0 0 1 0 1;2 2 0 0 2]
x[:,:,2,1] = [1 1 0 2 1;0 2 1 0 1;0 0 1 2 0;1 1 0 0 0;0 2 1 1 1]
x[:,:,3,1] = [0 2 1 0 0;2 0 0 1 1;2 2 1 2 1;1 1 1 0 1;0 2 2 2 2]

kern = zeros((3,3,3,2))
kern[:,:,1,1] = [1 -1 0;-1 -1 1;0 1 -1]
kern[:,:,2,1] = [-1 0 0;1 1 -1;1 -1 1]
kern[:,:,3,1] = [-1 -1 0;0 1 1;-1 1 1]

kern[:,:,1,2] = [0 0 -1;1 1 1;1 0 1]
kern[:,:,2,2] = [1 1 1;0 1 -1;1 1 -1]
kern[:,:,3,2] = [1 0 1;1 -1 0;1 1 0]

l.kern = (kern)
l.bias[1] = 1
l.bias[2] = 0

y = forward(l,x)
y_expected = zeros((3,3,2,1))
y_expected[:,:,1,1] = [9.0 3.0 3.0; 4.0 0.0 4.0; 0.0 1.0 1.0]
y_expected[:,:,2,1] = [1.0 6.0 8.0; 1.0 6.0 4.0; 5.0 4.0 3.0]
@test y == y_expected

dldy = ones(y_expected)
dldx = backward(l,dldy)
println(dldx)

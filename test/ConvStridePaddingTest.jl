include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")

batch_size = 1
input_size = (5,5,3)

l = CaffeConv(Dict("batch_size" => batch_size, "input_size" => input_size), 2, (3,3); padding=1, stride=2)

# x = zeros((3,3,3,1))
# x[:,:,1,1] = [1 2 3;0 0 0;0 0 0]
# x[:,:,2,1] = [0 0 0;4 5 6;0 0 0]
# x[:,:,3,1] = [1 2 3;0 0 0;7 8 9]
#
# kern = zeros((3,3,3,1))
# kern[:,:,1,1] = [1 0 0;0 1 0;0 0 1]
# kern[:,:,2,1] = [2 0 0;0 2 0;0 0 2]
# kern[:,:,3,1] = [3 0 0;0 3 0;0 0 3]

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

println(kern-flip(flip(kern)))
l.kern = (kern)
l.bias[1] = 1
l.bias[2] = 0

y = forward(l,x)
println(y)

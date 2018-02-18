include("../src/NN.jl")

using NN
using Base.Test

function testMaxPool(x, y, dldy, dldx, kernel, stride; alpha = 1.)
    l0 = InputLayer(size(x))
    l  = MaxPool(l0, (kernel, kernel), (stride, stride))

    # Testing forwarding
    @test forward(l, x) == y

    #Testing back propagations
    @test backward(l, dldy) == dldx
end

# First Test
println("Unit test 1...")
x_channel = reshape(1:16,4,4)

x = zeros(4,4,3,1)
for i = 1:3
    x[:,:,i,1] = x_channel
end

y_channel = [6 14; 8 16]
y = zeros(2,2,3,1)
for i = 1:3
    y[:,:,i,1] = y_channel
end

dldy_channel = [3 4; -1 -2]
dldy = zeros(2,2,3,1)
for i = 1:3
    dldy[:,:,i,1] = dldy_channel
end

dldx_channel = [0 0 0 0; 0 3 0 4; 0 0 0 0; 0 -1 0 -2]
dldx = zeros(4,4,3,1)
for i = 1:3
    dldx[:,:,i,1] = dldx_channel
end

testMaxPool(x, y, dldy, dldx, 2, 2)
println("test 1 passed.\n")

# Second Test
println("Unit test 2...")
x2_channel = reshape(1:9,3,3)
x2 = zeros(3,3,3,1)
for i = 1:3
    x2[:,:,i,1] = x2_channel
end

y2_channel = [5 8; 6 9]
y2 = zeros(2,2,3,1)
for i = 1:3
    y2[:,:,i,1] = y2_channel
end

dldy2_channel = [1 -1; 2 -2]
dldy2 = zeros(2,2,3,1)
for i = 1:3
    dldy2[:,:,i,1] = dldy2_channel
end

dldx2_channel = [0 0 0; 0 1 -1; 0 2 -2]
dldx2 = zeros(3,3,3,1)
for i = 1:3
    dldx2[:,:,i,1] = dldx2_channel
end

testMaxPool(x2, y2, dldy2, dldx2, 2, 2)
println("test 2 passed.\n")

# # Third test
# Second Test
println("Unit test 3...")
x3_channel = [7 1 9; 5 3 8; 6 4 0]
x3 = zeros(3,3,3,1)
for i = 1:3
    x3[:,:,i,1] = x3_channel
end

y3_channel = [7 9; 6 0]
y3 = zeros(2,2,3,1)
for i = 1:3
    y3[:,:,i,1] = y3_channel
end

dldy3_channel = [3 -1; 1 2]
dldy3 = zeros(2,2,3,1)
for i = 1:3
    dldy3[:,:,i,1] = dldy3_channel
end

dldx3_channel = [3 0 -1; 0 0 0; 1 0 2]
dldx3 = zeros(3,3,3,1)
for i = 1:3
    dldx3[:,:,i,1] = dldx3_channel
end

testMaxPool(x3, y3, dldy3, dldx3, 2, 2)
println("test 3 passed.\n")

# Third Test
println("Unit test 4...")
x_channel = reshape(1:16,4,4)

x = zeros(4,4,3,1)
for i = 1:3
    x[:,:,i,1] = x_channel
end

y_channel = [11 15; 12 16]
y = zeros(2,2,3,1)
for i = 1:3
    y[:,:,i,1] = y_channel
end

dldy_channel = [3 4; -1 -2]
dldy = zeros(2,2,3,1)
for i = 1:3
    dldy[:,:,i,1] = dldy_channel
end

dldx_channel = [0 0 0 0; 0 0 0 0; 0 0 3 4; 0 0 -1 -2]
dldx = zeros(4,4,3,1)
for i = 1:3
    dldx[:,:,i,1] = dldx_channel
end

testMaxPool(x, y, dldy, dldx, 3, 1)
println("test 4 passed.\n")

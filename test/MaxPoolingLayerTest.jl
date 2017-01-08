include("../src/layers/LayerBase.jl")
include("../src/layers/MaxPoolingLayer.jl")
using Base.Test



l1 = MaxPoolingLayer((2,2))
function beforeTest1(l)
    init(l, nothing, Dict{String, Any}("batch_size" => 1, "input_size" => [3,4,4])) #output size should be 3 * 2 * 2
end

l2 = MaxPoolingLayer((2,2)) 
function beforeTest2(l)
    init(l, nothing, Dict{String, Any}("batch_size" => 1, "input_size" => [3,3,3])) #output size should be 3 * 2 * 2
end


function testMaxPoolingLayer(l, x, y, dldy, dldx)
    # Testing forwarding
    @test forward(l,x) == y
    @test l.x == x
    print(l.max_idx[1,:,:,:])

    
    #Testing back propagation
    @test backward(l,dldy) == dldx
    @test l.dldy == dldy
    @test l.dldx == dldx
end

# First Test
println("Unit test 1...")
x_channel = reshape(1:16,4,4)
x = zeros(1,3,4,4)
for i = 1:3
    x[1,i,:,:] = x_channel
end

y_channel = [6 14; 8 16]
y = zeros(1,3,2,2)
for i = 1:3
    y[1,i,:,:] = y_channel
end

dldy_channel = [3 4; -1 -2]
dldy = zeros(1,3,2,2)
for i = 1:3
    dldy[1,i,:,:] = dldy_channel
end

dldx_channel = [0 0 0 0; 0 3 0 4; 0 0 0 0; 0 -1 0 -2]
dldx = zeros(1,3,4,4)
for i = 1:3
    dldx[1,i,:,:] = dldx_channel
end

beforeTest1(l1)
testMaxPoolingLayer(l1, x, y, dldy, dldx)
println("test 1 passed.\n")

# Second Test
println("Unit test 2...")
x2_channel = reshape(1:9,3,3)
x2 = zeros(1,3,3,3)
for i = 1:3
    x2[1,i,:,:] = x2_channel
end

y2_channel = [5 8; 6 9]
y2 = zeros(1,3,2,2)
for i = 1:3
    y2[1,i,:,:] = y2_channel
end

dldy2_channel = [1 -1; 2 -2]
dldy2 = zeros(1,3,2,2)
for i = 1:3
    dldy2[1,i,:,:] = dldy2_channel
end

dldx2_channel = [0 0 0; 0 1 -1; 0 2 -2]
dldx2 = zeros(1,3,3,3)
for i = 1:3
    dldx2[1,i,:,:] = dldx2_channel
end

beforeTest2(l2)
testMaxPoolingLayer(l2, x2, y2, dldy2, dldx2)
println("test 2 passed.\n")

# # Third test
# println("Unit test 3...")
# w2 = [2. 3.; 3. 2.]
# b2 = [1. 0;]
# x  = [1. 1;]
# y  = [6. 5;]
# dldy = [1. 2;]
# dldx = [8. 7;]
# gw2 = [1. 1; 2. 2;]
# gb2 = [1. 2;]
# beforeTest(l2)
# testDenseLayerOneVector(w2, b2, x, y, dldy, dldx, gw2, gb2)
# println("test 3 passed.\n")

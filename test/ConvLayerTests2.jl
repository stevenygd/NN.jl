include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConvLayer.jl")

# println(sum(forward(l,X)))

bsize= 1
l = CaffeConvLayer(1,(3,3))
# X = rand(5, 5, 3, bsize)
X = rand(5, 5, 1, bsize)
img1 = [
    1 3 2 0 2;
    1 0 2 3 3;
    0 2 3 0 1;
    3 0 0 2 0;
    1 0 2 0 1;
]

img2 = [
    3  4  0  1  1;
    1  -3 0  2  -3;
    -3 2  4  -3 0;
    0  2  -3 -3 -4;
    3  4  -2 0  -3;
]

img3 = [
    -1 0  -2 -1 4;
    3  -3 1  -1 0;
    0  -3 0  -4 0;
    2  -3 -2 -4 -4;
    -3 2  0  0  -2;
]

X[:,:,1,1] = img1
# X[:,:,2,1] = img2
# X[:,:,3,1] = img3

# init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 3)))
init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 1)))

l.kern[:,:,1,1] = [
    1 0 0;
    0 0 0;
    0 0 0
]

# l.kern[:,:,2,1] = [
#     0 0 0;
#     0 0 1;
#     0 0 0
# ]
#
# l.kern[:,:,3,1] = [
#     0 0 0;
#     0 1 0;
#     0 0 0
# ]

y = forward(l,X)
println(y)
# println(img1[1:3,1:3] + img2[2:4,3:5] + img3[2:4,2:4])
dldx = backward(l,y)
k_grad, b_grad = getGradient(l)
println("Backward:$(dldx)")
println("Gradient:$(k_grad)")

my_grad = [
    sum(img1[1:3,1:3].*y) sum(img1[1:3,2:4].*y) sum(img1[1:3,3:5].*y);
    sum(img1[2:4,1:3].*y) sum(img1[2:4,2:4].*y) sum(img1[2:4,3:5].*y);
    sum(img1[3:5,1:3].*y) sum(img1[3:5,2:4].*y) sum(img1[3:5,3:5].*y);
]
println("My Gradient:$(my_grad)")

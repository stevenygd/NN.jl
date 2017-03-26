include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConvLayer.jl")
include("gradient_test.jl")
using Base.Test

bsize= 500
l = CaffeConvLayer(32,(3,3))
init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => (27, 27, 3)))
X = rand(27, 27, 3,  bsize)
Y = rand(25, 25, 32, bsize)
function f_kernel(k)
    l.kern = k
    return sum(forward(l,X))
end
backward(l,ones(size(forward(l, X))))
g, _ = getGradient(l)
k = copy(l.kern)
v1 = f_kernel(k)
k[10] +=  1e-4
v2 = f_kernel(k)
println("$(v1) $(v2) $(abs(v2-v1))")

anl_g, err = gradient_check(f_kernel, g, k)
println("Relative error: $(err) $(mean(abs(anl_g))) $(mean(abs(g)))")
# @test_approx_eq_eps anl_g g 1e-4*max(abs(anl_g))
@test_approx_eq_eps err 0. 1e-4
println("[PASS] convolution gradient check test.")

# println(f_kernel(k))
# println(sum(forward(l,X)))

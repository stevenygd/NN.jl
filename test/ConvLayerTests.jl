include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConvLayer.jl")
include("gradient_test.jl")
using Base.Test

bsize= 500
inp_size = (5,5,3)
f = 3
ksize = (3,3)
l = CaffeConvLayer(f,ksize)
init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => inp_size))
X = rand(inp_size[1], inp_size[2], inp_size[3],  bsize)
println("X statistics: $(mean(abs(X))) $(maximum(X)) $(minimum(X))")

function f_kernel(k)
    l.kern = k
    return sum(forward(l,X))
end
backward(l,ones(size(forward(l, X))))
g, _ = getGradient(l)
k = copy(l.kern)

anl_g, err = gradient_check(f_kernel, g, k)
println("Relative error: $(err) $(mean(abs(anl_g))) $(mean(abs(g)))")

@test_approx_eq_eps anl_g g (1e-8*maximum(abs(anl_g)))
println("[PASS] convolution gradient check test with analytic gradients (1e-8)")

@test_approx_eq_eps err 0. 1e-4
println("[PASS] convolution gradient check test (1e-4)")
@test_approx_eq_eps err 0. 1e-5
println("[PASS] convolution gradient check test.(1e-5)")
@test_approx_eq_eps err 0. 1e-6
println("[PASS] convolution gradient check test.(1e-6)")
@test_approx_eq_eps err 0. 1e-7
println("[PASS] convolution gradient check test.(1e-7)")
@test_approx_eq_eps err 0. 1e-6
println("[PASS] convolution gradient check test.(1e-8)")

# println(f_kernel(k))
# println(sum(forward(l,X)))

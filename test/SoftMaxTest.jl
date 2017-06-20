include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMax.jl")
using Base.Test
using ForwardDiff

l = SoftMax();
s = x->x->exp(x)./sum(exp(x))
g = x->ForwardDiff.gradient(s, x)

function before(l)
    init(l, nothing, Dict{String, Any}("batch_size" => 1, "input_size" => [3]))
end

function testSoftMax(x::Array{Float64}, y::Array{Float64}, dldy::Array{Float64}, dldx::Array{Float64}; alpha = .001)
  # Testing forwarding
  before(l)
  @test_approx_eq forward(l,x) exp(x)./sum(exp(x))
  @test_approx_eq backward(l,dldy) dldx

end

# First Test
println("Unit test 1...")
x = reshape([1.0, 2.0, 3.0;], 1,3)
dldy = reshape([1. 1. 1.;], 1,3)
testSoftMax(x, s(x), dldy, g(x))
println("test 1 passed.\n")

include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMax.jl")
using Base.Test
using ForwardDiff

l = SoftMax();
s(x::Vector) = exp(x)./sum(exp(x))
g(x::Vector) = ForwardDiff.jacobian(s,x)

function before(l,x)
    m,n = size(x)
    init(l, nothing, Dict{String, Any}("batch_size" => m, "input_size" => [n]))
end

function testSoftMax(x::Array{Float64}, y::Array{Float64}, dldy::Array{Float64}, dldx::Array{Float64}; alpha = 1.)
  # Testing forwarding
  before(l,x)
  f = vec(forward(l,x))
  b = vec(backward(l,dldy))
  @test_approx_eq_eps norm(f-y) 0 alpha
  @test_approx_eq_eps norm(b-dldx) 0 alpha

end


# Basic Test
println("Unit test 1...")
x = rand(1,3)
dldy = ones(1,3)
dldy_v = [1., 1., 1.]
testSoftMax(x, s(vec(x)), dldy, g(vec(x))*dldy_v)
println("Basic test passed.\n")

# Number < 30 with 10 classes
println("Unit test 2...")
x = 30*rand(1,10)
dldy = ones(1,10)
dldy_v = vec(dldy)
testSoftMax(x, s(vec(x)), dldy, g(vec(x))*dldy_v)
println("Complicated test passed.\n")

# Big number test
println("Unit test 3...")
x = 692*rand(1, 100)
dldy = ones(1,100)
dldx = g(vec(x))*vec(dldy)
testSoftMax(x, s(vec(x)), dldy, dldx)
println("Big number test passed.\n")

# Big big number big classes test
println("Unit test 4...")
x = 678*rand(1,1000)
x_v = vec(x)
dldy = ones(1,1000)
dldy_v = vec(dldy)
testSoftMax(x, s(x_v), dldy, g(x_v)*dldy_v)
println("Big big number test passed.\n")

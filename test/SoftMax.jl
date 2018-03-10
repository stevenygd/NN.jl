include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMax.jl")
using Base.Test
using ForwardDiff

s(x::Vector) = exp.(x)./sum(exp.(x))
g(x::Vector) = ForwardDiff.jacobian(s,x)

function testSoftMaxOneVector(x::Array{Float64}, y::Array{Float64}, dldy::Array{Float64}, dldx::Array{Float64}; alpha=1.0)
  m,n = size(x)
  l = SoftMax(Dict{String, Any}("batch_size" => 0, "input_size" => n))
  f = vec(forward(l,x))
  b = vec(backward(l,dldy))
  @test norm(f-y) ≈ 0 atol=alpha
  @test norm(b-dldx) ≈ 0 atol=alpha

end

function s_m(x::Array{Float64,2})
  f = zeros(size(x))
  for i=1:size(x)[1]
    f[i,:] = s(vec(x[i,:]))
  end
  f
end

function g_m(x::Array{Float64,2}, dldy::Array{Float64,2})
  f = zeros(size(x))
  for i=1:size(x)[1]
    f[i,:] = g(vec(x[i,:]))*dldy[i,:]
  end
  f
end

function testSoftMaxGeneral(x::Array{Float64}, dldy::Array{Float64}; alpha=1.0)
  m,n = size(x)
  l = SoftMax(Dict{String, Any}("batch_size" => 0, "input_size" => n))
  
  @assert size(x) == size(dldy)

  f = forward(l,x)
  fs = s_m(x)
  b = backward(l,dldy)
  bg = g_m(x,dldy)

  @test norm(f-fs) ≈ 0 atol=alpha
  @test norm(b-bg) ≈ 0 atol=alpha

end

# Basic Test
println("Unit test 1...")
x = rand(1,3)
dldy = ones(1,3)
dldy_v = [1., 1., 1.]
testSoftMaxOneVector(x, s(vec(x)), dldy, g(vec(x))*dldy_v)
println("Basic test passed.\n")

# Number < 30 with 10 classes
println("Unit test 2...")
x = 30*rand(1,10)
dldy = ones(1,10)
dldy_v = vec(dldy)
testSoftMaxOneVector(x, s(vec(x)), dldy, g(vec(x))*dldy_v)
println("Complicated test passed.\n")

# Big number test
println("Unit test 3...")
x = 692*(2*rand(1, 100)-1)
dldy = rand(1,100)
dldx = g(vec(x))*vec(dldy)
testSoftMaxOneVector(x, s(vec(x)), dldy, dldx)
println("Big number test passed.\n")

# Big big number big classes test
println("Unit test 4...")
x = 678*rand(1,1000)
x_v = vec(x)
dldy = rand(1,1000)
dldy_v = vec(dldy)
testSoftMaxOneVector(x, s(x_v), dldy, g(x_v)*dldy_v)
println("Big big number test passed.\n")

println("Unit test 5...")
x = 57*(2*rand(500, 10)-1)
dldy = rand(500,10)
testSoftMaxGeneral(x, dldy)
println("Multiple instances test passed.\n")

println("Unit test 6...")
x = 235*(2*rand(1000, 100)-1)
dldy = rand(1000,100)
testSoftMaxGeneral(x, dldy)
println("Big multiple instances test passed.\n")

println("Unit test 7...")
x = 476*(2*rand(5000, 5000)-1)
dldy = rand(5000,5000)
testSoftMaxGeneral(x, dldy)
println(令人疯狂的 test passed.\n")

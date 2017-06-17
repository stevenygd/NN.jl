include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/SoftMax.jl")
include("../src/layers/CrossEntropyLoss.jl")
using Base.Test

smaxcross = SoftMaxCrossEntropyLoss()
smax = SoftMax()
cross = CrossEntropyLoss()


function before(smaxcross, smax, cross, dict)
    init(smaxcross, nothing, dict)
    init(smax     , nothing, dict)
    init(cross    , nothing, dict)
end

function benchmmark(smaxcross, smax, cross, batch_size, input_size; alpha = 1.)
  # batch_size = 10
  # input_size = 5
  before(smaxcross, smax, cross, Dict{String, Any}("batch_size" => batch_size, "input_size" => [input_size]))
  x = rand(batch_size, input_size)
  label = rand(0:input_size-1, batch_size,1)
  # benchmark for original SoftMaxCrossEntropy
  tic()

  # l1, p1 = @time forward(smaxcross,x,label)
  l1, p1 = forward(smaxcross,x,label)

  time = toq()
  println("old forward uses:  ", time)
  tic()

  # d1 = @time backward(smaxcross,label)
  d1 = backward(smaxcross,label)

  time = toq()
  println("old backward uses: ", time)

  # benchmark for newly implemented softmax & CrossEntropyLoss
  tic()

  # l2, p2 = @time forward(cross, forward(smax,x), label)
  l2, p2 = forward(cross, forward(smax,x), label)

  time = toq()
  println("new forward uses:  ", time)
  tic()

  # d2 = @time backward(smax, backward(cross, label))
  d2 = backward(smax, backward(cross, label))

  time = toq()
  println("new backward uses: ", time)

  @test_approx_eq l1 l2
  @test_approx_eq p1 p2
  println("Correctness test passed")
end

for i=1:20
  benchmmark(smaxcross, smax, cross, 1000, 100; alpha = 1.)
end

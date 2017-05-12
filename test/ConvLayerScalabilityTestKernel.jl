include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConvLayer.jl")

using PyPlot

# scalibility for kernel
forward_times, backward_times, gradient_times = Float64[], Float64[], Float64[]
forward_alloc, backward_alloc, gradient_alloc = Float64[], Float64[], Float64[]
w,h,c,f = 32, 32, 64, 64
b   = 128
X = rand(w,h,c,b)
K = 32
for k = 1:K
    l = CaffeConvLayer(f,(k,k))
    init(l, nothing, Dict{String, Any}("batch_size" => b, "input_size" => (w,h,c)))

    _,t,a,_,_ = @timed Y = forward(l, X)
    append!(forward_times,t)
    append!(forward_alloc,a)

    _,t,a,_,_ = @timed X = backward(l, Y)
    append!(backward_times,t)
    append!(backward_alloc,a)

    _,t,a,_,_ = @timed G = getGradient(l)
    append!(gradient_times,t)
    append!(gradient_alloc,a)

end

figure(figsize=(12,6))
plot(1:K,forward_times, label="forward")
plot(1:K,backward_times,label="backward")
plot(1:K,gradient_times,label="gradient")
xlabel("Kernal size")
ylabel("Time for one pass")
title("Time v.s. Kernel Size")
legend(loc="upper left",fancybox="true")
show()

figure(figsize=(12,6))
plot(1:K,forward_alloc/1024^2, label="forward")
plot(1:K,backward_alloc/1024^2,label="backward")
plot(1:K,gradient_alloc/1024^2,label="gradient")
xlabel("Kernal size")
ylabel("Memory Allocation for one pass (MB)")
title("Allocation v.s. Kernel Size")
legend(loc="upper left",fancybox="true")
show()

println(forward_alloc)
println(backward_alloc)
println(gradient_alloc)

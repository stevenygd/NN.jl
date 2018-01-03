include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")

using PyPlot

# scalibility for batch_size
forward_times, backward_times, gradient_times = Float64[], Float64[], Float64[]
forward_alloc, backward_alloc, gradient_alloc = Float64[], Float64[], Float64[]
w,h,c,f = 32, 32, 64, 64
B   = 302
k = 5
for b = 32:10:B
    l = CaffeConv(Dict{String, Any}("batch_size" => b, "input_size" => (w,h,c)),f,(k,k))
    X = rand(w,h,c,b)

    println("Working on batch_size:$(b)")
    _,t,a,_,_ = @timed Y = forward(l, X)
    append!(forward_times,t)
    append!(forward_alloc,a)
    println("\tForward :$(t) $(a)");

    _,t,a,_,_ = @timed X = backward(l, Y)
    append!(backward_times,t)
    append!(backward_alloc,a)
    println("\tBackward :$(t) $(a)");

    _,t,a,_,_ = @timed G = getGradient(l)
    append!(gradient_times,t)
    append!(gradient_alloc,a)
    println("Gradients :$(t) $(a)");

end

figure(figsize=(12,6))
plot(32:10:B,forward_times, label="forward")
plot(32:10:B,backward_times,label="backward")
plot(32:10:B,gradient_times,label="gradient")
xlabel("Batch Size")
ylabel("Time for one pass")
title("Time v.s. Batch Size")
legend(loc="upper left",fancybox="true")
show()

figure(figsize=(12,6))
plot(32:10:B,forward_alloc/1024^2, label="forward")
plot(32:10:B,backward_alloc/1024^2,label="backward")
plot(32:10:B,gradient_alloc/1024^2,label="gradient")
xlabel("Batch Size")
ylabel("Memory Allocation for one pass (MB)")
title("Allocation v.s. Batch Size")
legend(loc="upper left",fancybox="true")
show()

println(forward_alloc)
println(backward_alloc)
println(gradient_alloc)

# scalibility for channels

# scalibility for image size

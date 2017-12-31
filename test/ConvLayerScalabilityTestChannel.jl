include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")

using PyPlot

# scalibility for channels
forward_times, backward_times, gradient_times = Float64[], Float64[], Float64[]
forward_alloc, backward_alloc, gradient_alloc = Float64[], Float64[], Float64[]
w,h,C,f = 32, 32, 130, 64
b = 64
k = 5
for c = 1:2:C
    l = CaffeConv(f,(k,k))
    init(l, nothing, Dict{String, Any}("batch_size" => b, "input_size" => (w,h,c)))
    X = rand(w,h,c,b)

    println("Working on channel:$(c)")
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
plot(1:2:C,forward_times, label="forward")
plot(1:2:C,backward_times,label="backward")
plot(1:2:C,gradient_times,label="gradient")
xlabel("Channel Size")
ylabel("Time for one pass")
title("Time v.s. Channel Size")
legend(loc="upper left",fancybox="true")
show()

figure(figsize=(12,6))
plot(1:2:C,forward_alloc/1024^2, label="forward")
plot(1:2:C,backward_alloc/1024^2,label="backward")
plot(1:2:C,gradient_alloc/1024^2,label="gradient")
xlabel("Channel Size")
ylabel("Memory Allocation for one pass (MB)")
title("Allocation v.s. Channel Size")
legend(loc="upper left",fancybox="true")
show()

# scalibility for image size

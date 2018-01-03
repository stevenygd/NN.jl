include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")

using PyPlot

# scalibility for image size
forward_times, backward_times, gradient_times = Float64[], Float64[], Float64[]
forward_alloc, backward_alloc, gradient_alloc = Float64[], Float64[], Float64[]
I,c,f = 256, 130, 64
b = 64
k = 5
for i = 32:32:I
    w=h=i
    l = CaffeConv(f,(k,k))
    init(l, nothing, Dict{String, Any}("batch_size" => b, "input_size" => (w,h,c)))
    X = rand(w,h,c,b)

    println("Working on image size:$(w) $(h)")
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
plot((32:32:I).^2,forward_times, label="forward")
plot((32:32:I).^2,backward_times,label="backward")
plot((32:32:I).^2,gradient_times,label="gradient")
xlabel("Image Size")
ylabel("Time for one pass")
title("Time v.s. Image Size")
legend(loc="upper left",fancybox="true")
show()

figure(figsize=(12,6))
plot((32:32:I).^2,forward_alloc/1024^2, label="forward")
plot((32:32:I).^2,backward_alloc/1024^2,label="backward")
plot((32:32:I).^2,gradient_alloc/1024^2,label="gradient")
xlabel("Image Size")
ylabel("Memory Allocation for one pass (MB)")
title("Allocation v.s. Image Size")
legend(loc="upper left",fancybox="true")
show()

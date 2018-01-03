include("../src/layers/LayerBase.jl")

function test_performance(l, lname; bsize=500, input_size=(1,27,27), output_size=(25,25))
    function X()
        c, w, h = input_size
        return rand(bsize, c, w, h)
    end

    function Y()
        w, h = output_size
        return rand(bsize, l.filter, w, h)
    end

    println("First time (compiling...)")
    init(l, nothing, Dict{String, Any}("batch_size" => bsize, "input_size" => input_size))

    # First pass through the function, so it's compiled
    forward(l,X())
    backward(l,Y())
    getGradient(l)

    println("Time for one forward pass.")
    x = X()
    @time forward(l,x)

    println("Time for one backward pass.")
    y = Y()
    @time backward(l,y)

    println("Time for one gradients")
    @time getGradient(l)

    println()
    n = 10
    println("Comprehensive profiling ($(n) batches of all things):")
    t = 0
    for _ = 1:n
        x = X()
        t += @elapsed forward(l,x)
        y = Y()
        t += @elapsed backward(l,y)
        t += @elapsed getGradient(l)
    end
    println("$(lname) used $(t) seconds to run $(n) passes")
end

include("../src/layers/Conv.jl")
test_performance(Conv(32,(3,3)), "Conv")

include("../src/layers/FlatConv.jl")
test_performance(FlatConv(32,(3,3)), "FlatConv")

include("../src/layers/CaffeConv.jl")
test_performance(CaffeConv(32,(3,3)), "CaffeConv")


# include("../layers/MultiThreadedConv.jl")

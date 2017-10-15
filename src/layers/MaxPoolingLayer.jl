# include("LayerBase.jl")

type MaxPoolingLayer <: RegularizationLayer
    parents  :: Array{Layer}
    children :: Array{Layer}

    has_init :: Bool

    # Parameters
    size     :: Tuple{Int, Int}     # (pool_width, pool_height)
    stride   :: Int

    # Input output place holders
    x        :: Array{Float64, 4}   # (batch_size, channel, width,     height)
    dldx     :: Array{Float64, 4}   # (batch_size, channel, width,     height)
    y        :: Array{Float64, 4}   # (batch_size, channel, out_width, out_height)
    dldy     :: Array{Float64, 4}   # (batch_size, channel, out_width, out_height)

    # Index of the maximum
    # (batch_size, channel, out_width, out_height)
    max_idx  :: Array{Tuple{Int, Int}, 4}

    function MaxPoolingLayer(size::Tuple{Int,Int};stride = 1)
        @assert stride == 1 # TODO: doesn't allow other stride yet
        return new(Layer[], Layer[], false, size, stride, zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), Array{Tuple{Int, Int}}(1,1,1,1))
    end

    function MaxPoolingLayer(prev::Union{Layer,Void}, size::Tuple{Int,Int}, config::Dict{String, Any}; stride = 1)
        layer = new(Layer[], Layer[], false, size, stride, zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), Array{Tuple{Int, Int}}(1,1,1,1))
        init(layer, prev, config)
        layer
    end
end

function computeOutputSize(l::MaxPoolingLayer, input_size::Tuple)
    w, h, c, b = input_size
    x, y = l.size
    s = l.stride
    return (Int(ceil(w/x)), Int(ceil(h/y)), c, b)
end

function init(l::MaxPoolingLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    """
    Initialize the Convolutional layers. Preallocate all the memories.
    """
	if !isa(p,Void)
        l.parents = [p]
        push!(p.children, l)
    end

    if p == nothing
        @assert length(config["input_size"]) == 3
        batch_size = config["batch_size"]
        w, h, c    = config["input_size"]
        input_size = (w, h, c, batch_size)
    else
        input_size = getOutputSize(p)
    end
    @assert length(input_size) == 4
    output_size  = computeOutputSize(l, input_size)

    # initialize input/output
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    l.max_idx = Array{Tuple{Int, Int}}(output_size)

    l.has_init = true
end

function update(l::MaxPoolingLayer, input_size::Tuple;)
    # assert: only change the batch sizes
    @assert length(input_size) == 4
    @assert input_size[1:end-1] == size(l.x)[1:end-1]

    b = input_size[4]
    output_size = size(l.y)
    output_size = (output_size[1], output_size[2], output_size[3], b)

    # Relinitialize input and output
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    l.max_idx = Array{Tuple{Int,Int}}(output_size)

    # println("MaxPooling Layer update shape:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

function forward(l::MaxPoolingLayer; kwargs...)
    l.y = forward(l, l.children[1].y)
end

function forward(l::MaxPoolingLayer, x::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    batch_size, img_depth = size(l.x, 4), size(l.x, 3)
    in_w, in_h = size(l.x, 1), size(l.x, 2)
    sw, sh = l.size
    for b = 1:batch_size
    for c = 1:img_depth
    for w = 1:size(l.y,1)
    for h = 1:size(l.y,2)
        start_w, start_h   = (w-1)*sw+1, (h-1)*sh+1
        end_w,   end_h     = min(in_w, sw*w), min(in_h, sh*h)
        view_w, view_h     = end_w - start_w + 1, end_h - start_h + 1

        l.y[w,h,c,b], i    = findmax(view(l.x, start_w:end_w, start_h:end_h, c, b))
        l.max_idx[w,h,c,b] = (start_w + (i-1)%view_w, start_h + Int(floor((i-1)/view_w)))
        # x,y = l.max_idx[w,h,c,b]
        # if x > end_w || y > end_h
        #     println("$(start_w), $(end_w)")
        #     println("$(start_h), $(end_h)")
        #     println("$(l.y[w,h,c,b]), $(i)")
        #     println("$((i-1)%sw), $(Int(floor((i-1)/sh)))")
        #     println(l.max_idx[w,h,c,b])
        # end
        # ix, iy = l.max_idx[w,h,c,b]
    end; end; end; end
    return l.y
end

function backward(l::MaxPoolingLayer, dldy::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    l.dldy = dldy
    scale!(l.dldx, 0.)#clear l.dldx
    batch_size, img_depth = size(l.x, 4), size(l.x, 3)
    for b = 1:batch_size
    for c = 1:img_depth
    for w = 1:size(l.y,1)
    for h = 1:size(l.y,2)
        x, y = l.max_idx[w,h,c,b]
        # print("x:$(x),y:$(y) coming from w:$(w), h:$(h)")
        l.dldx[x,y,c,b] = l.dldy[w,h,c,b]
    end; end; end; end
    return l.dldx
end

# l = MaxPoolingLayer((3,3))
# X = rand(64, 3, 31, 31)
# Y = rand(64, 3, 11, 11)
#
# println("First time (compiling...)")
# @time init(l, nothing, Dict{String, Any}("batch_size" => 64, "input_size" => (3,31,31)))
# @time forward(l,X)
# @time backward(l,Y)
# @time getGradient(l)

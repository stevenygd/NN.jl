type MaxPool <: RegularizationLayer
    base     :: LayerBase

    # Parameters
    kernel_size  :: Tuple{Int, Int}     # (pool_width, pool_height)
    stride       :: Tuple{Int, Int}

    # Input output place holders
    x        :: Array{Float64, 4}   # (batch_size, channel, width,     height)
    dldx_cache     :: Array{Float64, 4}   # (batch_size, channel, width,     height)
    dldy     :: Array{Float64, 4}   # (batch_size, channel, out_width, out_height)

    # Index of the maximum
    # (batch_size, channel, out_width, out_height)
    max_idx  :: Array{Tuple{Int, Int}, 4}

    function MaxPool(prev::Layer, kernel_size::Tuple{Int,Int}; stride=kernel_size, kwargs...)
        if stride == Void
            stride = kernel_size
        end
        layer = new(LayerBase(), kernel_size, stride, zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), Array{Tuple{Int, Int}}(1,1,1,1))
        connect(layer, [prev])
        init(layer, getOutputSize(prev); kwargs...)
        layer
    end

    function MaxPool(config::Dict{String,Any}, kernel_size::Tuple{Int,Int}; stride=kernel_size, kwargs...)
        if l.stride == Void
            l.stride = l.kernel_size
        end
        layer = new(LayerBase(), kernel_size, stride, zeros(1,1,1,1), zeros(1,1,1,1),
                   zeros(1,1,1,1), zeros(1,1,1,1), Array{Tuple{Int, Int}}(1,1,1,1))
        input_size = (config["input_size"], config["batch_size"])
        init(layer, input_size; kwargs...)
        layer
    end
end

function computeOutputSize(l::MaxPool, input_size::Tuple)
    w, h, c, b = input_size
    x, y = l.kernel_size
    sw, sh = l.stride
    return (Int(ceil((w-x)/sw))+1, Int(ceil((h-y)/sh))+1, c, b)
end

function init(l::MaxPool, input_size::Tuple; kwargs...)
    """
    Initialize the Convolutional layers. Preallocate all the memories.
    """
    @assert length(input_size) == 4
    output_size  = computeOutputSize(l, input_size)
    # Default Stride is kernel size


    # initialize input/output
    l.x    = Array{Float64}(input_size)
    l.dldx_cache = Array{Float64}(input_size)
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    l.base.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    l.max_idx = Array{Tuple{Int, Int}}(output_size)
end

function update(l::MaxPool, input_size::Tuple;)
    # assert: only change the batch sizes
    @assert length(input_size) == 4
    @assert input_size[1:end-1] == size(l.x)[1:end-1]

    b = input_size[4]
    output_size = size(l.base.y)
    output_size = (output_size[1], output_size[2], output_size[3], b)

    # Relinitialize input and output
    l.x    = Array{Float64}(input_size)
    l.dldx_cache = Array{Float64}(input_size)
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    l.base.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    l.max_idx = Array{Tuple{Int,Int}}(output_size)

    # println("MaxPooling Layer update shape:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

function forward(l::MaxPool, x::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    batch_size, img_depth = size(l.x, 4), size(l.x, 3)
    in_w, in_h = size(l.x, 1), size(l.x, 2)
    sw, sh = l.stride
    kw, kh = l.kernel_size
    for b = 1:batch_size
    for c = 1:img_depth
    for w = 1:size(l.base.y,1)
    for h = 1:size(l.base.y,2)
    start_w, start_h   = (w-1)*sw+1, (h-1)*sh+1
    end_w,   end_h     = min(in_w, kw*w), min(in_h, kh*h)
    view_w, view_h     = end_w - start_w + 1, end_h - start_h + 1

    l.base.y[w,h,c,b], i    = findmax(view(l.x, start_w:end_w, start_h:end_h, c, b))
    l.max_idx[w,h,c,b] = (start_w + (i-1)%view_w, start_h + Int(floor((i-1)/view_w)))

    end;end;end;end
    return l.base.y
end

function backward(l::MaxPool, dldy::Union{SubArray{Float64,4},Array{Float64,4}}; kwargs...)
    l.dldy = dldy
    fill!(l.dldx_cache, 0.)
    batch_size, img_depth = size(l.x, 4), size(l.x, 3)
    for b = 1:batch_size
    for c = 1:img_depth
    for w = 1:size(l.base.y,1)
    for h = 1:size(l.base.y,2)
        x, y = l.max_idx[w,h,c,b]
        # print("x:$(x),y:$(y) coming from w:$(w), h:$(h)")
        l.dldx_cache[x,y,c,b] = l.dldy[w,h,c,b]
    end; end; end; end
    parent_id = l.base.parents[1].base.id
    l.base.dldx[parent_id] = l.dldx_cache
end

# l = MaxPool((3,3))
# X = rand(64, 3, 31, 31)
# Y = rand(64, 3, 11, 11)
#
# println("First time (compiling...)")
# @time init(l, nothing, Dict{String, Any}("batch_size" => 64, "input_size" => (3,31,31)))
# @time forward(l,X)
# @time backward(l,Y)
# @time getGradient(l)

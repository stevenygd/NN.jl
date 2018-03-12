# Abtract out DropOut
type Flatten <: UtilityLayer
    base     :: LayerBase

    x           :: Array{Float64,4}
    dldy        :: Array{Float64,2}

    function Flatten(prev::Layer)
        layer = new(LayerBase(), Array{Float64}(1,1,1,1), Array{Float64}(1,1))
        connect(layer, [prev])
        init(layer, getOutputSize(prev))
        layer
    end

    function Flatten(config::Dict{String,Any})
        layer = new(LayerBase(), Array{Float64}(1,1,1,1), Array{Float64}(1,1))
        out_size = (config["input_size"], config["batch_size"])
        init(layer, prev, config)
        layer
    end
end

function init(l::Flatten, input_size::Tuple; kwargs...)
    @assert length(input_size) == 4
    w, h, c, b  = input_size
    output_size = b, c*w*h
    l.x    = Array{Float64}(input_size)
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    l.base.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
end

function update(l::Flatten, input_size::Tuple;)
    # only allow to change the batch size
    @assert length(input_size) == 4 && input_size[1:3] == size(l.x)[1:3]
    # b,x,y,z = input_size
    x,y,z,b = input_size
    output_size = b,x*y*z
    l.base.dldx[l.base.parents[1].base.id] = Array{Float64}(input_size)
    l.x    = Array{Float64}(input_size)
    l.base.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    # println("Flatten update:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

function forward(l::Flatten, x::Union{SubArray{Float64,4},Array{Float64,4}}; deterministics=false)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    permutedims!(l.base.y, reshape(l.x, (size(l.x,1)*size(l.x,2)*size(l.x,3), size(l.x,4))), [2,1])
    return l.base.y
end

# Donot annotate DLDY since it could be subarray
function backward(l::Flatten, dldy::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    l.dldy = dldy
    parent_id = l.base.parents[1].base.id
    l.base.dldx[parent_id] = reshape(permutedims(l.dldy, [2,1]), size(l.x))
end

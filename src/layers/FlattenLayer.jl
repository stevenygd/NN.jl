# Abtract out DropOut
# include("LayerBase.jl")
type FlattenLayer <: UtilityLayer
    has_init    :: Bool
    x           :: Array{Float64,4}
    y           :: Array{Float64,2}
    dldx        :: Array{Float64,4}
    dldy        :: Array{Float64,2}

    function FlattenLayer()
        return new(false, Array{Float64}(1,1,1,1), Array{Float64}(1,1),
                   Array{Float64}(1,1,1,1), Array{Float64}(1,1))
    end
end

function init(l::FlattenLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    if p != nothing
        input_size = getOutputSize(p)
    else
        batch_size = config["batch_size"]
        w,h,c      = config["input_size"]
        input_size = w, h, c, batch_size
    end
    @assert length(input_size) == 4
    w, h, c, b  = input_size
    output_size = b, c*w*h
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    l.has_init = true
end

function update(l::FlattenLayer, input_size::Tuple;)
    # only allow to change the batch size
    @assert length(input_size) == 4 && input_size[1:3] == size(l.x)[1:3]
    # b,x,y,z = input_size
    x,y,z,b = input_size
    output_size = b,x*y*z
    l.x    = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.y    = Array{Float64}(output_size)
    l.dldy = Array{Float64}(output_size)
    println("FlattenLayer update:\n\tInput:$(input_size)\n\tOutput:$(output_size)")
end

function forward(l::FlattenLayer, x::Union{SubArray{Float64,4},Array{Float64,4}}; deterministics=false)
    if size(x) != size(l.x)
        update(l, size(x))
    end
    l.x = x
    permutedims!(l.y, reshape(l.x, (size(l.x,1)*size(l.x,2)*size(l.x,3), size(l.x,4))), [2,1])
    return l.y
end

# Donot annotate DLDY since it could be subarray
function backward(l::FlattenLayer, dldy::Union{SubArray{Float64,2},Array{Float64,2}}; kwargs...)
    l.dldy = dldy
    l.dldx = reshape(permutedims(l.dldy, [2,1]), size(l.x))
    return l.dldx
end

# l = FlattenLayer()
# x = rand(64,3,30,30)
# y = rand(64,2700)
#
# forward(l,x)
# backward(l,y)

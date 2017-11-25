abstract type ANN end
abstract type Layer end
abstract type LearnableLayer <: Layer end
abstract type Nonlinearity   <: Layer end
abstract type LossCriteria   <: Layer end
abstract type DataLayer      <: Layer end
abstract type RegularizationLayer <: Layer end
abstract type UtilityLayer <: Layer end

type LayerBase
    parents  :: Array{Layer}
    children :: Array{Layer}
    id  :: Base.Random.UUID

    y        :: Array{Float64}
    dldx     :: Dict{Base.Random.UUID, Array{Float64}}

    function LayerBase()
        return new(Layer[], Layer[], Base.Random.uuid4(),
            Float64[], Dict{Base.Random.UUID,Array{Float64}}())
    end
end

function connect(l::Layer, parents::Array{<:Layer})
    for x ∈ parents
        push!(l.base.parents, x)
        push!(x.base.children, l)
    end
end

function forward(l ::Layer; kwargs...)
	forward(l, l.base.parents[1].base.y; kwargs...)
end

function backward(l ::Layer; kwargs...)
    DLDY = sum(map(x -> x.base.dldx[l.base.id], l.base.children))
    backward(l, DLDY; kwargs...)
end

StaticLayer = Union{Nonlinearity, DataLayer, RegularizationLayer, UtilityLayer}

function getGradient(l::StaticLayer)
    return nothing
end

function getParam(l::StaticLayer)
    return nothing
end

function setParam!(l::StaticLayer, theta)
    return nothing
end

function getVelocity(l::StaticLayer)
    return nothing
end

function getInputSize(l::Layer)
    return size(l.base.x)
end

function getOutputSize(l::Layer)
    return size(l.base.y)
end

function getNumParams(l::Layer)
    return 0
end

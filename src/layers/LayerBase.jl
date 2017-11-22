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
    has_init :: Bool
    id  :: Base.Random.UUID

    y        :: Array{Float64}
    dldx     :: Dict{Base.Random.UUID, Array{Float64}}

    function LayerBase()
        return new(Layer[], Layer[], false, Base.Random.uuid4(),
            Float64[], Float64[], Float64[], Dict{Base.Random.UUID,Array{Float64}}())
    end
end

function connect(l::Layer, parents::Array{<:Layer})
    # if !isa(p,Void)
    #     l.parents = [p]
    #     push!(p.children, l)
    # end
    for x ∈ parents
        push!(l.base.parents, x)
        push!(x.base.children, l)
    end
end

StaticLayer = Union{Nonlinearity, DataLayer, RegularizationLayer, UtilityLayer}

function getGradient(l::StaticLayer)
    return nothing
end

function getParam(l::StaticLayer)
    return nothing
end

function setParam!(l::StaticLayer, theta)
    nothing
end

function getVelocity(l::StaticLayer)
    return nothing
end

function getInputSize(l::Layer)
    if !l.base.has_init
        println("Warning: layer $(l) hasn't been initizalized. But input shapes wanted.")
    end
    return size(l.base.x)
end

function getOutputSize(l::Layer)
    if !l.base.has_init
        println("Warning: layer $(l) hasn't been initizalized. But output shapes wanted.")
    end
    return size(l.base.y)
end

function getNumParams(l::Layer)
    return 0
end

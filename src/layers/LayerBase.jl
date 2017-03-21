abstract ANN
abstract Layer
abstract LearnableLayer <: Layer
abstract Nonlinearity   <: Layer
abstract LossCriteria   <: Layer
abstract DataLayer      <: Layer
abstract RegularizationLayer <: Layer
abstract UtilityLayer <: Layer

StaticLayer = Union{Nonlinearity, DataLayer, RegularizationLayer, UtilityLayer}


function getGradient(l::StaticLayer)
    return Array[]
end

function getParam(l::StaticLayer)
    return Array[]
end

function setParam!(l::StaticLayer, theta)
    nothing
end

function getVelocity(l::StaticLayer)
    return Array[]
end

function getInputSize(l::Layer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initizalized. But input shapes wanted.")
    end
    return size(l.x)
end

function getOutputSize(l::Layer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initizalized. But output shapes wanted.")
    end
    return size(l.y)
end

function getNumParams(l::Layer)
    return 1
end

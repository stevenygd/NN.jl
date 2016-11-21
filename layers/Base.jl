abstract NN
abstract Layer
abstract LearnableLayer <: Layer
abstract Nonlinearity   <: Layer
abstract LossCriteria   <: Layer
abstract NoiseLayer     <: Layer

StaticLayer = Union{Nonlinearity, NoiseLayer}


function gradient(l::StaticLayer)
    0
end

function getParam(l::StaticLayer)
    0
end

function setParam!(l::StaticLayer, theta)
    nothing
end

function getVelocity(l::StaticLayer)
    0
end

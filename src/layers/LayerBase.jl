abstract ANN
abstract Layer
abstract LearnableLayer <: Layer
abstract Nonlinearity   <: Layer
abstract LossCriteria   <: Layer
abstract NoiseLayer     <: Layer
abstract DataLayer      <: Layer

StaticLayer = Union{Nonlinearity, NoiseLayer, DataLayer}


function getGradient(l::StaticLayer)
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

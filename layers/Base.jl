abstract NN
abstract Layer
abstract Nonlinearity  <: Layer
abstract LossCriteria  <: Layer

function gradient(l::Nonlinearity)
    return 0
end

function getParam(l::Nonlinearity)
    return 0
end

function setParam!(l::Nonlinearity, theta)
    nothing
end

function getLDiff(l::Nonlinearity)
    0
end

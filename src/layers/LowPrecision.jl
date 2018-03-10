type LowPrecision <: Wrapper
    wb :: WrapperBase
    acc :: type

    function LowPrecision(layer::Layer, acc::type)
        return new(wb(layer), acc)
    end
end

function forward()
end

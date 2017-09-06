type Graph <: ANN
    layer :: Layer

    function Graph(layer::Layer)
        # Initialize all the layers
        return new(layers)
    end
end

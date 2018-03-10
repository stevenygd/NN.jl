type Addition <: Layer
    base    :: LayerBase

    function Addition(prevs::Array{<:Layer}; kwargs...)
        layer =  new(LayerBase())
        init(layer, prevs; kwargs...)
        layer
    end
end

function init(l::Addition, ps::Union{Array{<:Layer}}; kwargs...)
    @assert length(ps) ≥ 1
    out_size = getOutputSize(ps[1])
    for p in ps
        @assert getOutputSize(p) == out_size
        push!(p.base.children, l)
        push!(l.base.parents, p)
    end

    l.base.y = Array{Float64}(out_size)
    foreach(x -> l.base.dldx[x.base.id] = Array{Float64}(out_size), l.base.parents)

end

function update(l::Addition, input_size::Tuple;)
    l.base.y = Array{Float64}(input_size)
    foreach(x -> l.base.dldx[x.base.id] = Array{Float64}(input_size), l.base.parents)
end

function forward(l::Addition;kwargs...)
    xs = [l.base.y for l in l.base.parents]
    l.base.y = zeros(l.base.y)
    for i=1:size(xs)[1]
        broadcast!(+, l.base.y, l.base.y, xs[i])
    end
end

function backward(l::Addition;kwargs...)
    DLDY = sum(map(x -> x.base.dldx[l.base.id], l.base.children))
    foreach(x -> l.base.dldx[x.base.id] = DLDY, l.base.parents)
end

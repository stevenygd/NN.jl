type RMSPropOptimizer
    net     :: SequentialNet
    cache   :: Any
    lr_base :: Float64
    alpha   :: Float64

    function RMSPropOptimizer(net::SequentialNet; lr_base=0.001, alpha::Float64 = 0.9)
        cache = []
        for i = 1:length(net.layers)
            layer = net.layers[i]
            param = getParam(layer)
            if param == nothing
                push!(cache, nothing)
            else
                c = []
                for j = 1:length(param)
                    push!(c, zeros(size(param[j])))
                end
                push!(cache, c)
            end
        end;
        return new(net, cache, lr_base, alpha);
    end
end

function optimize(this::RMSPropOptimizer, batch_X, batch_Y)
    loss, pred = forward(this.net, batch_X, batch_Y)
    backward(this.net, batch_Y)

    for i = 1:length(this.net.layers)
        layer = this.net.layers[i]
        param = getParam(layer)
        if param == nothing
            continue # not a learnable layer
        end

        grad  = getGradient(layer)
        for j = 1:length(param)
            c = this.cache[i][j]
            p = param[j]
            g = grad[j]
            @assert size(c) == size(p) && size(c) == size(g)
            c = c * this.alpha + g.^2 * (1 - this.alpha)
            p = p - this.lr_base * g ./ (sqrt.(c) + 1e-8)
            this.cache[i][j] = c
            param[j] =    p
        end
        setParam!(layer, param)
    end
    return loss, pred
end;

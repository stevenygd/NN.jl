type CdamOptimizer
    net     :: SequentialNet
    w_m     :: Any
    w_v     :: Any
    base_lr :: Float64
    beta_1  :: Float64
    beta_2  :: Float64
    iter    :: Int
    re_iter :: Int

    function CdamOptimizer(net::SequentialNet;
                           base_lr::Float64=0.001,
                           beta_1::Float64=0.9, beta_2::Float64=0.999,
                           re_iter::Int=10)

        w_m, w_v = [], []
        for i = 1:(re_iter + 1)
            m_t, v_t = [], []
            for i = 1:length(net.layers)
               param = getParam(net.layers[i])
               if param == nothing
                   push!(m_t, nothing)
                   push!(v_t, nothing)
               else
                   c_1, c_2 = [], []
                   for j = 1:length(param)
                       push!(c_1, zeros(size(param[j])))
                       push!(c_2, zeros(size(param[j])))
                   end
                   push!(m_t, c_1)
                   push!(v_t, c_2)
               end;
            end;
            push!(w_m, m_t)
            push!(w_v, v_t)
        end
        return new(net, w_m, w_v, base_lr, beta_1, beta_2, 1, re_iter)
    end
end

function optimize(this::CdamOptimizer, batch_X, batch_Y)

    loss, pred = forward(this.net, batch_X, batch_Y)
    backward(this.net, batch_Y)

    curr_iter = (this.iter - 1) % this.re_iter + 1
    for i = 1:length(this.net.layers)
        layer = this.net.layers[i]
        param = getParam(layer)
        if param == nothing
            continue # not a learnable layer
        end

        grad  = getGradient(layer)

        for j = 1:length(param)
            p = param[j]
            g = grad[j]
            g2 = g.^2

            last_m = this.w_m[curr_iter][i][j]
            last_v = this.w_v[curr_iter][i][j]

            this.w_m[curr_iter][i][j] = g
            this.w_v[curr_iter][i][j] = g2

            # Window moving average
            if curr_iter == this.iter # not yet set finding the whole window
                m = (this.w_m[end][i][j] * (this.iter - 1) + g)  / this.iter
                v = (this.w_v[end][i][j] * (this.iter - 1) + g2) / this.iter
            else
                m = this.w_m[end][i][j] + (g  - last_m) / this.re_iter
                v = this.w_v[end][i][j] + (g2 - last_v) / this.re_iter
            end

            # Store the newest moving avg
            this.w_m[end][i][j] = m
            this.w_v[end][i][j] = v

            # Prepare gradients update
            p = p - this.base_lr * m ./ sqrt.(v + 1e-8)
            param[j] = p
        end
        setParam!(layer, param)
    end

    this.iter += 1;
    return loss, pred
end

using Base.Test

function gradient_check(f, g, w, d=1e-5, tol=0.01, nsamples=1)
    """Checking whether gradient [g] is correctly computed for function [f]
    Inputs:
        [f] the function that computes the gradients
        [w] the input weight point we are testing at
        [g] the computed gradient (f'(w) = g)

    Output:
        [anl_g] element-wise analytical gradient estimated using [(f(w+d) - f(w-d))/2d]
        [err_r] mean element-wise relative error by [|anl_g - g|/max(|anl_g|, |g|)]
    """
    @assert size(g) == size(w)
    anl_g = zeros(size(g))
    w_pos = copy(w)
    w_neg = copy(w)

    for i = 1:Int(length(w))
        @inbounds w_pos[i] = w[i] + d
        @inbounds w_neg[i] = w[i] - d
        v1 = f(w_pos)
        v2 = f(w_neg)
        tmp_anl_g = (v1 - v2) / (2. * d)
        # println("Analytical Gradient: $(tmp_anl_g) $(v1) $(v2) $(d)")
        @inbounds w_pos[i] = w[i]
        @inbounds w_neg[i] = w[i]
        @inbounds anl_g[i] = tmp_anl_g
    end;
    err_r = abs.(anl_g .- g) ./ (max.(abs.(anl_g), abs.(g)) + 1e-10)
    return anl_g, mean(err_r)
end;

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
        @inbounds w_pos[i] = w[i]
        @inbounds w_neg[i] = w[i]
        @inbounds anl_g[i] = tmp_anl_g
    end;
    err_r = abs(anl_g .- g) ./ max(abs(anl_g), abs(g))
    return anl_g, mean(err_r)
end;

# Test for gradient test
function f1(w)
    return (w[1] - w[2])^2
end
w1 = [3. 4.]
g1 = [-2. 2.] # [2(w1[1] - w[2]), -2(w1[1] - w[2])]
anl_g1, err_r1 = gradient_check(f1,g1,w1)
@test_approx_eq_eps anl_g1 g1 1e-8
@test_approx_eq_eps err_r1 0. 1e-7
println("[PASS] Gradient check test 1 pass.")

function f2(w)
    return mean((w[:,1] + w[:,2]).^2)
end
w2 = randn(100,2)
g2 = [(w2[:,1] + w2[:,2]) (w2[:,1] + w2[:,2])] ./ 50.
println(size(w2), size(g2))
anl_g2, err_r2 = gradient_check(f2,g2,w2)
@test_approx_eq_eps anl_g2 g2 1e-8
@test_approx_eq_eps err_r2 0. 1e-7
println("[PASS] Gradient check test 2 pass.")

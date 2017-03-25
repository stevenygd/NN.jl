
function gradient_check(f,g,w, d=1e-5, tol=0.01)
    """Checking whether gradient [g] is correctly computed for function [f]
    Inputs:
        [f] the function that computes the gradients
        [w] the input weight point we are testing at
        [g] the computed gradient (f'(w) = g)

    Output:
        [anl_g] element-wise analytical gradient estimated using [(f(w+d) - f(w-d))/2d]
        [err_r] element-wise relative error by [|anl_g - g|/max(anl_g, g)]
    """
    
end;

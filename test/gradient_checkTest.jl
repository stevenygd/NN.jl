includ("gradient_check.jl")
function f1(w)
    return (w[1] - w[2])^2
end
w1 = [3. 4.]
g1 = [-2. 2.] # [2(w1[1] - w[2]), -2(w1[1] - w[2])]
anl_g1, err_r1 = gradient_check(f1,g1,w1)
@test anl_g1 ≈ g1 atol=1e-8
@test err_r1 ≈ 0. atol=1e-7
println("[PASS] Gradient check test 1 pass.")

function f2(w)
    return mean((w[:,1] + w[:,2]).^2)
end
w2 = randn(100,2)
g2 = [(w2[:,1] + w2[:,2]) (w2[:,1] + w2[:,2])] ./ 50.
println(size(w2), size(g2))
anl_g2, err_r2 = gradient_check(f2,g2,w2)
@test anl_g2 ≈ g2 atol=1e-8
@test err_r2 ≈ 0. atol=1e-7
println("[PASS] Gradient check test 2 pass.")

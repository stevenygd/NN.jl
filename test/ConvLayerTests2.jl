include("../src/layers/LayerBase.jl")
include("../src/layers/CaffeConv.jl")
using Base.Test
# println(sum(forward(l,X)))

function test1()
    bsize= 1
    l = CaffeConv(Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 1)), 1,(3,3))
    X = rand(5, 5, 1, bsize)

    img1 = [
        1 3 2 0 2;
        1 0 2 3 3;
        0 2 3 0 1;
        3 0 0 2 0;
        1 0 2 0 1;
    ]

    X[:,:,1,1] = img1

    l.kern[:,:,1,1] = [
        1 0 0;
        0 0 0;
        0 0 0
    ]

    y = forward(l,X)
    @test img1[1:3,1:3] ≈ y
    println("[Test 1.1][Pass] Forward Unit Test 1 Passes.")

    # println(y)
    # println(img1[1:3,1:3] + img2[2:4,3:5] + img3[2:4,2:4])
    dldx = backward(l,y)
    # TODO: backward test?

    k_grad, b_grad = getGradient(l)
    # println("Gradient:$(k_grad)")

    my_grad = [
        sum(img1[1:3,1:3].*y) sum(img1[1:3,2:4].*y) sum(img1[1:3,3:5].*y);
        sum(img1[2:4,1:3].*y) sum(img1[2:4,2:4].*y) sum(img1[2:4,3:5].*y);
        sum(img1[3:5,1:3].*y) sum(img1[3:5,2:4].*y) sum(img1[3:5,3:5].*y);
    ]
    @test my_grad ≈ k_grad
    println("[Test 1.3][Pass] Gradient Unit Test 1 Passed.")
    # println("My Gradient:$(my_grad)")
end

function test2()
    ################################################################################
    #  Test 2 Starts                                                               #
    ################################################################################
    bsize= 1
    l = CaffeConv(Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 3)),1,(3,3))
    X = rand(5, 5, 3, bsize)
    img1 = [
        1 3 2 0 2;
        1 0 2 3 3;
        0 2 3 0 1;
        3 0 0 2 0;
        1 0 2 0 1;
    ]
    img2 = [
        3  4  0  1  1;
        1  -3 0  2  -3;
        -3 2  4  -3 0;
        0  2  -3 -3 -4;
        3  4  -2 0  -3;
    ]

    img3 = [
        -1 0  -2 -1 4;
        3  -3 1  -1 0;
        0  -3 0  -4 0;
        2  -3 -2 -4 -4;
        -3 2  0  0  -2;
    ]
    X[:,:,1,1] = img1
    X[:,:,2,1] = img2
    X[:,:,3,1] = img3

    l.kern[:,:,1,1] = [
        0 0 0;
        0 0 0;
        1 0 0
    ]

    l.kern[:,:,2,1] = [
        0 0 0;
        0 0 1;
        0 0 0
    ]

    l.kern[:,:,3,1] = [
        0 0 0;
        0 1 0;
        0 0 0
    ]
    y = forward(l,X)
    ans_y = img1[3:5, 1:3] + img2[2:4,3:5] + img3[2:4,2:4]
    @test y ≈ ans_y
    println("[Test 2.1][Pass] Pass Forward Unit Test 2.")

    dldx = backward(l, y)
    g, _ = getGradient(l)
    g_ans = zeros(size(g))
    for c = 1:3
        g_ans[:,:,c,1] = [
            sum(X[:,:,c,1][1:3,1:3].*y) sum(X[:,:,c,1][1:3,2:4].*y) sum(X[:,:,c,1][1:3,3:5].*y);
            sum(X[:,:,c,1][2:4,1:3].*y) sum(X[:,:,c,1][2:4,2:4].*y) sum(X[:,:,c,1][2:4,3:5].*y);
            sum(X[:,:,c,1][3:5,1:3].*y) sum(X[:,:,c,1][3:5,2:4].*y) sum(X[:,:,c,1][3:5,3:5].*y);
        ]
    end
    @test g ≈ g_ans
    println("[Test 2.3][Pass] Pass Gradient Unit Test 2.")
end

function test3()
    ################################################################################
    #  Test 3 Starts                                                               #
    ################################################################################
    bsize= 1
    l = CaffeConv(Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 1)),3,(3,3))
    X = rand(5, 5, 1, bsize)
    img1 = [
        3 4 0 1 1;
        1 3 0 2 3;
        3 2 4 3 0;
        0 2 3 3 4;
        3 4 2 0 3;
    ]

    X[:,:,1,1] = img1

    l.kern[:,:,1,1] = [
        0 0 0;
        1 0 0;
        0 0 0
    ]

    l.kern[:,:,1,2] = [
        0 0 0;
        0 0 0;
        0 1 0
    ]

    l.kern[:,:,1,3] = [
        0 1 0;
        0 0 0;
        0 0 0
    ]

    y = forward(l,X)
    ans_y = zeros(size(y))
    for f = 1:3
        ans_y[:,:,f,1] = [
            sum(l.kern[:,:,1,f].*img1[1:3,1:3]) sum(l.kern[:,:,1,f].*img1[1:3,2:4]) sum(l.kern[:,:,1,f].*img1[1:3,3:5]);
            sum(l.kern[:,:,1,f].*img1[2:4,1:3]) sum(l.kern[:,:,1,f].*img1[2:4,2:4]) sum(l.kern[:,:,1,f].*img1[2:4,3:5]);
            sum(l.kern[:,:,1,f].*img1[3:5,1:3]) sum(l.kern[:,:,1,f].*img1[3:5,2:4]) sum(l.kern[:,:,1,f].*img1[3:5,3:5]);
        ]
    end
    @test y ≈ ans_y
    println("[Test 3.1][Pass] Pass Forward Unit Test 3.")

    dldx = backward(l, y)
    g, _ = getGradient(l)
    g_ans = zeros(size(g))
    for f = 1:3
        g_ans[:,:,1,f] = [
            sum(X[:,:,1,1][1:3,1:3].*y[:,:,f,1]) sum(X[:,:,1,1][1:3,2:4].*y[:,:,f,1]) sum(X[:,:,1,1][1:3,3:5].*y[:,:,f,1]);
            sum(X[:,:,1,1][2:4,1:3].*y[:,:,f,1]) sum(X[:,:,1,1][2:4,2:4].*y[:,:,f,1]) sum(X[:,:,1,1][2:4,3:5].*y[:,:,f,1]);
            sum(X[:,:,1,1][3:5,1:3].*y[:,:,f,1]) sum(X[:,:,1,1][3:5,2:4].*y[:,:,f,1]) sum(X[:,:,1,1][3:5,3:5].*y[:,:,f,1]);
        ]
    end
    @test g ≈ g_ans
    println("[Test 2.3][Pass] Pass Gradient Unit Test 3.")
end

function test4()
    ################################################################################
    #  Test 4 Starts                                                               #
    ################################################################################
    bsize= 1
    l = CaffeConv(Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 3)),3,(3,3))
    X = rand(5, 5, 3, bsize)
    X[:,:,1,1] = [
        3 -4 0 1 -1;
        -1 3 0 2 3;
        3 2 -4 3 0;
        0 2 3 -3 4;
        3 4 2 0 -3;
    ]

    X[:,:,2,1] = [
        3  4  0  1  1;
        1  -3 0  2  -3;
        -3 2  4  -3 0;
        0  2  -3 -3 -4;
        3  4  -2 0  -3;
    ]

    X[:,:,3,1] = [
        -1 0  -2 -1 4;
        3  -3 1  -1 0;
        0  -3 0  -4 0;
        2  -3 -2 -4 -4;
        -3 2  0  0  -2;
    ]

    for f = 1:3
        for c = 1:3
            l.kern[:,:,c,f] = [
               1 2 3;
               4 5 6;
               7 8 9.
            ] * 10^((f-1)*3 + c-1)
        end
    end

    y = forward(l,X)
    ans_y = zeros(size(y))
    for f = 1:3
        for c = 1:3
            ans_y[:,:,f,1] += [
                sum(l.kern[:,:,c,f].*X[:,:,c,1][1:3,1:3]) sum(l.kern[:,:,c,f].*X[:,:,c,1][1:3,2:4]) sum(l.kern[:,:,c,f].*X[:,:,c,1][1:3,3:5]);
                sum(l.kern[:,:,c,f].*X[:,:,c,1][2:4,1:3]) sum(l.kern[:,:,c,f].*X[:,:,c,1][2:4,2:4]) sum(l.kern[:,:,c,f].*X[:,:,c,1][2:4,3:5]);
                sum(l.kern[:,:,c,f].*X[:,:,c,1][3:5,1:3]) sum(l.kern[:,:,c,f].*X[:,:,c,1][3:5,2:4]) sum(l.kern[:,:,c,f].*X[:,:,c,1][3:5,3:5]);
            ]
        end;
    end
    @test y ≈ ans_y
    println("[Test 4.1][Pass] Pass Forward Unit Test 4.")

    dldx = backward(l, y)
    g, _ = getGradient(l)
    g_ans = zeros(size(g))
    for f = 1:3
        for c = 1:3
            g_ans[:,:,c,f] = [
                sum(X[:,:,c,1][1:3,1:3].*y[:,:,f,1]) sum(X[:,:,c,1][1:3,2:4].*y[:,:,f,1]) sum(X[:,:,c,1][1:3,3:5].*y[:,:,f,1]);
                sum(X[:,:,c,1][2:4,1:3].*y[:,:,f,1]) sum(X[:,:,c,1][2:4,2:4].*y[:,:,f,1]) sum(X[:,:,c,1][2:4,3:5].*y[:,:,f,1]);
                sum(X[:,:,c,1][3:5,1:3].*y[:,:,f,1]) sum(X[:,:,c,1][3:5,2:4].*y[:,:,f,1]) sum(X[:,:,c,1][3:5,3:5].*y[:,:,f,1]);
            ]
        end
    end
    @test g ≈ g_ans
    println("[Test 4.3][Pass] Pass Gradient Unit Test 4.")
end

function test5()
    ################################################################################
    #  Test 5 Starts                                                               #
    ################################################################################
    bsize= 3
    l = CaffeConv(Dict{String, Any}("batch_size" => bsize, "input_size" => (5, 5, 3)),3,(3,3))
    X = rand(5, 5, 3, bsize)
    for b = 1:3
        for c = 1:3
            X[:,:,c,b] = rem.(rand(Int, 5,5), 5)
        end
    end

    for f = 1:3
        for c = 1:3
            l.kern[:,:,c,f] = [
               1 2 3;
               4 5 6;
               7 8 9.
            ] * 10^((f-1)*3 + c-1)
        end
    end

    y = forward(l,X)
    ans_y = zeros(size(y))
    for b = 1:3
        for f = 1:3
            for c = 1:3
                ans_y[:,:,f,b] += [
                    sum(l.kern[:,:,c,f].*X[:,:,c,b][1:3,1:3]) sum(l.kern[:,:,c,f].*X[:,:,c,b][1:3,2:4]) sum(l.kern[:,:,c,f].*X[:,:,c,b][1:3,3:5]);
                    sum(l.kern[:,:,c,f].*X[:,:,c,b][2:4,1:3]) sum(l.kern[:,:,c,f].*X[:,:,c,b][2:4,2:4]) sum(l.kern[:,:,c,f].*X[:,:,c,b][2:4,3:5]);
                    sum(l.kern[:,:,c,f].*X[:,:,c,b][3:5,1:3]) sum(l.kern[:,:,c,f].*X[:,:,c,b][3:5,2:4]) sum(l.kern[:,:,c,f].*X[:,:,c,b][3:5,3:5]);
                ]
            end;
        end
    end
    @test y ≈ ans_y
    println("[Test 5.1][Pass] Pass Forward Unit Test 5.")

    dldx = backward(l, y)
    g, _ = getGradient(l)
    g_ans = zeros(size(g))
    for f = 1:3
        for c = 1:3
            for b=1:3
                g_ans[:,:,c,f] += [
                    sum(X[:,:,c,b][1:3,1:3].*y[:,:,f,b]) sum(X[:,:,c,b][1:3,2:4].*y[:,:,f,b]) sum(X[:,:,c,b][1:3,3:5].*y[:,:,f,b]);
                    sum(X[:,:,c,b][2:4,1:3].*y[:,:,f,b]) sum(X[:,:,c,b][2:4,2:4].*y[:,:,f,b]) sum(X[:,:,c,b][2:4,3:5].*y[:,:,f,b]);
                    sum(X[:,:,c,b][3:5,1:3].*y[:,:,f,b]) sum(X[:,:,c,b][3:5,2:4].*y[:,:,f,b]) sum(X[:,:,c,b][3:5,3:5].*y[:,:,f,b]);
                ]
            end
        end
    end
    @test g ≈ g_ans
    println("[Test 5.3][Pass] Pass Gradient Unit Test 5.")
end

test1()
test2()
test3()
test4()
test5()

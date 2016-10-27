using Base.Test

tests = ["FCLayerTest",
         "ReLuTest",
         "SequentialNetTest"]

println("Running tests:")

for t in tests
    println(" * $(t)")
    include("$(t).jl")
end
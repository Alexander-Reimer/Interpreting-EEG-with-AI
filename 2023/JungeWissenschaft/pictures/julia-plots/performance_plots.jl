using Pkg
Pkg.activate("../../../.")
include("../../../main.jl")
using PyPlot
using BenchmarkTools

function compare_performance_one_vs_two()
    AI.hyper_params = AI.Args(0.001, 200, 1, 100, [1, 2];
        cuda = false, one_out = true)
    global times_one = []
    times_two = []
    @timed println("oinhu")
    for i = 1:10
        stats = @timed AI.train(true)
        push!(times_one, stats.time)
    end
end
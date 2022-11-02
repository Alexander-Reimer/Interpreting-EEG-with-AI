"""
Alternative pruning test not using pruning.jl --  maybe useful for looking for mistakes in puning.jl.
"""

using PyPlot
using Statistics: mean
pygui(true)
fig = figure("Test")
clf()

include("BCI.jl")
# using .BCI

include("config.jl") # defines variable "config"

data = BCI.get_data(config)
# model = BCI.load_model("model-logging/conv_sentdex_2022-10-08_11:43:28/")
model = BCI.new_model(config)

# colors = ["blue", "red", "green", "orange", "yellow", "black", "grey"]
@info "Everything prepared!"


function add2line!(line, y)
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if isempty(ydata)
        line.set_data(
            [1],
            [y]
            )
    else
        line.set_data(
            [xdata..., xdata[end] + 1],
            [ydata..., y]
            )
    end
end
    
function foo(i)
    train_avg, test_avg = BCI.get_avg_activations(model, data)
    diff = train_avg - test_avg
    diff = [abs.(x) for x in diff]
    push!(diffs, diff)

    for (line, y) in enumerate(maximum.(diff))
        add2line!(lines_max[line], y)
    end

    for (line, y) in enumerate(mean.(diff))
        add2line!(lines_mean[line], y)
    end

    for ax in [ax_max, ax_mean]
        ax.relim()
        ax.relim()
        ax.autoscale_view()
        ax.autoscale_view(scalex=true, scaley=false)
    end
    PyPlot.show()
end

function init()
    global ax_max, ax_mean = fig.subplots(2, 1, sharex=true)
    global lines
    for li in eachindex(model.model.layers)
        push!(lines_max, ax_max.plot([], [], label="Layer $li (Max)")[1])
        push!(lines_mean, ax_mean.plot([], [], label="Layer $li (Mean)", "--")[1])
    end
    # ax_max.legend()
    # ax_mean.legend()
end

function bar()
    global i, diffs
    
    foo(i)
    for it = (i+1):(i+5)
        global model, data
        model.epochs_goal += 3
        trainmode!(model.model)
        BCI.train!(model, data)
        testmode!(model.model)
        foo(it)
        println("Repeat $(it) done!")
    end
    i += 5
    tight_layout()
end

i = 0
diffs = []
lines_max = []
lines_mean = []

init()
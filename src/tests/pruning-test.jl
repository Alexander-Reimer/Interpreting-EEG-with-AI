"""
Plot activation differences while training.
"""

using PyPlot
using Statistics: mean
pygui(true)
fig = figure("Test")
clf()

include("../BCI.jl")
# using .BCI

include("../config.jl") # defines variable "config"

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
    fit_avg, _ = BCI.determine_overfit(model, data, BCI.average)
    fit_max, indexes = BCI.determine_overfit(model, data, maximum)
    push!(diffs, diff)

    for (line, y) in enumerate(fit_max)
        if indexes[line] != 0
            add2line!(lines_max[line], y)
        end
    end
    
    for (line, y) in enumerate(fit_avg)
        if indexes[line] != 0
            add2line!(lines_mean[line], y)
        end
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
    _, indexes = BCI.determine_overfit(model, data, BCI.average)

    for li in eachindex(model.model.layers)
        if indexes[li] == 0
            push!(lines_max, nothing)
            push!(lines_mean, nothing)
        else
            push!(lines_max, ax_max.plot([], [], label="Layer $li (Max)")[1])
            push!(lines_mean, ax_mean.plot([], [], label="Layer $li (Mean)", "--")[1])
        end
    end
    # ax_max.legend()
    # ax_mean.legend()
end

function bar()
    global i, diffs
    
    foo(i)
    for it = (i+1):(i+10)
        global model, data
        model.epochs_goal += 2
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
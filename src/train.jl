module AI

using PyCall, Flux, PyPlot
np = pyimport("numpy")
using Flux: crossentropy, train!, onecold
using BSON
include("common_functions.jl")

include("default_config.jl") # provide default options, don't change
include("config.jl") # overwrite default options, you just need to set the variables

function get_data(path)
    return np.load(path)
end

function get_all_data(folder)
    # Get data of all files in $folder
    # Return 3D-Array, (sample, channel, amplitude of frequency given by fft)
    files = readdir(folder)
    l = 0
    for file in files
        d = get_data(folder * file)
        l += size(d)[1]
    end

    data = zeros(Float32, l, 16, 60)
    i = 0
    index = 1

    for file in files
        #println(i)
        d = get_data(folder * file)
        i += 1
        data[index:index+size(d)[1]-1, :, :] = d
        index += size(d)[1]
    end
    return data
end

function get_formatted_data(path, output, test)
    # Transform data, add outputs
    # Add data X and Y to global variables X_traindata, X_testdata, Y_traindata, Y_testdata
    data = get_all_data(path)
    data = mapslices(rotr90, data, dims=[1, 3])

    if test
        global X_testdata[:, 1, :, index+1:index+size(data)[3]] = data
    else
        global X_traindata[:, 1, :, index+1:index+size(data)[3]] = data
    end

    for i = 1:size(data)[3]
        if test
            global Y_testdata[:, index+i] = output
        else
            global Y_traindata[:, index+i] = output
        end
    end
    global index += size(data)[3]
end

function check_NaN(model)
    # See if gradients contain NaNs
    ps = Flux.params(model)
    gs = Flux.gradient(ps) do
        loss(X_traindata, Y_traindata)
    end
    search_NaN = []
    for elements in gs
        push!(search_NaN, 1 ∈ isnan.(elements))
    end
    return search_NaN
end

function get_accuracy(model, data_x, data_y)
    total = 0
    correct = 0
    all_out = model(data_x)
    for i = 1:size(data_x)[4]
        est = all_out[:, i]
        if argmax(est) == argmax(data_y[:, i])
            correct += 1
        end
        total += 1
    end
    return correct / total
end

mutable struct Plot
    fig
    train_loss
    test_loss
    train_acc
    test_acc
end

function init_plot()
    if PLOT[1]
        fig = figure("Model performance history")
        train_loss = plot([], [])[1]
        test_loss = plot([], [])[1]
        train_acc = plot([], [])[1]
        test_acc = plot([], [])[1]
        global the_plot = Plot(fig, train_loss, test_loss, train_acc, test_acc)
    end
end

function advance_history()
    if isempty(x_history)
        push!(x_history, 0)
    else
        push!(x_history, x_history[end] + 1)
    end
    println("Iteration $(x_history[end]):")
    if HISTORY_TRAINLOSS[1] && mod(x_history[end], HISTORY_TRAINLOSS[2]) == 0
        push!(train_loss_history, loss(X_traindata, Y_traindata))
        println("   train loss: ", train_loss_history[end])
        if PLOT[1] && mod(x_history[end], PLOT[2]) == 0
            the_plot.train_loss.set_data(x_history, train_loss_history)
        end
    else
        push!(train_loss_history, nothing)
    end
    if HISTORY_TESTLOSS[1] && mod(x_history[end], HISTORY_TESTLOSS[2]) == 0
        push!(test_loss_history, loss(X_testdata, Y_testdata))
        println("   test loss: ", test_loss_history[end])
        if PLOT[1] && mod(x_history[end], PLOT[2]) == 0
            the_plot.test_loss.set_data(x_history, test_loss_history)
        end
    else
        push!(test_loss_history, nothing)
    end
    if HISTORY_TRAINACCURACY[1] && mod(x_history[end], HISTORY_TRAINACCURACY[2]) == 0
        push!(train_accuracy_history, accuracy(X_traindata, Y_traindata))
        println("   train accuracy: ", train_accuracy_history[end])
        if PLOT[1] && mod(x_history[end], PLOT[2]) == 0
            the_plot.train_accuracy.set_data(x_history, train_accuracy_history)
        end
    else
        push!(train_accuracy_history, nothing)
    end
    if HISTORY_TESTACCURACY[1] && mod(x_history[end], HISTORY_TESTACCURACY[2]) == 0
        push!(test_accuracy_history, accuracy(X_testdata, Y_testdata))
        println("   test accuracy: ", test_accuracy_history[end])
        if PLOT[1] && mod(x_history[end], PLOT[2]) == 0
            the_plot.test_accuracy.set_data(x_history, test_accuracy_history)
        end
    else
        push!(test_accuracy_history, nothing)
    end
    if PLOT[1]
        PyPlot.show()
    end
end

function init_model()
    if isempty(LOAD_PATH)
        global model = MODEL()

        global x_history = []
        global train_loss_history = []
        global test_loss_history = []
        global train_accuracy_history = []
        global test_accuracy_history = []

        advance_history()
    else
        load_model()
    end
    global ps = Flux.params(model)
end

# DATA
# *************************************************************************************************************************

num_outputs = length(TRAIN_DATA[1][2])

# Pre-initialise arrays to improve performance
global X_traindata = Array{Float32}(undef, MAX_FREQUENCY, 1, NUM_CHANNELS, NUM_SAMPLES_TRAIN)
global Y_traindata = Array{Float32}(undef, num_outputs, NUM_SAMPLES_TRAIN)

global X_testdata = Array{Float32}(undef, MAX_FREQUENCY, 1, NUM_CHANNELS, NUM_SAMPLES_TEST)
global Y_testdata = Array{Float32}(undef, num_outputs, NUM_SAMPLES_TEST)

# Populate arrays with eeg data
global index = 0
for (path, output) in TRAIN_DATA
    get_formatted_data(path, output, false)
end

global index = 0
for (path, output) in TEST_DATA
    get_formatted_data(path, output, true)
end

# Turn data into Flux DataLoaders
global train_data = Flux.Data.DataLoader((X_traindata, Y_traindata), batchsize=BATCH_SIZE, shuffle=true, partial=false)
global test_data = Flux.Data.DataLoader((X_testdata, Y_testdata), batchsize=BATCH_SIZE, shuffle=true, partial=false)

# Clear unnecessary data
# X_traindata = nothing
# Y_traindata = nothing
# X_testdata = nothing
# Y_testdata = nothing


# TRAINING
# *************************************************************************************************************************


loss(x, y) = LOSS(model(x), y)
opt = OPTIMIZER(LEARNING_RATE)

init_plot()
init_model()
for iteration = 1:ITERATIONS
    for (x, y) in train_data
        gs = Flux.gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end
    advance_history()
end

save_model()

println(model(X_traindata[:, :, :, 1:1]))

#=
data_ = [Array{Float32}(data_[i,:,:]) for i in 1:size(data_)[1]]
data = Vector{Matrix{Float32}}(undef, length(data_))
for i = 1:length(data_)
    data[i] = reshape(data_[i], 1, :)
end
println("transformed data", length(data))
=#

end #module
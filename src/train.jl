module AI

using PyCall, Flux, PyPlot, BSON, ProgressMeter, CUDA
using CUDA: CuIterator
np = pyimport("numpy")
using Flux: crossentropy, train!, onecold

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

function noise(x)
    # Function for gaussian random noise, from https://fluxml.ai/tutorials/2021/02/07/convnet.html
    return x .+ device(0.1f0 * randn(eltype(x), size(x)))
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

function loss_accuracy(type::Symbol)
    if type == :train
        return loss_accuracy(train_data, string(type))
    elseif type == :test
        return loss_accuracy(test_data, string(type))
    else
        @warn "Type $type unknown!"
    end
end

function loss_accuracy(data_loader::Flux.Data.DataLoader, name::String)
    total = 0
    accurate = 0
    l = Float32(0)
    i = 0
    # 
    @showprogress 2 "    Calculating $(name) performance..." for (x, y) in (data_loader)
        i += 1
        if mod(i, (1.0 / LOSS_ACCURACY_PORTION)) == 0
            x, y = x |> device, y |> device
            est = model(x)
            l += LOSS(est, y)
            same_result_bitarray = onecold(est) .== onecold(y)
            accurate += sum(same_result_bitarray)
            total += length(same_result_bitarray)
        end
    end
    return (loss=l / total, accuracy=accurate / total)
end

mutable struct Plot
    fig
    ax_loss
    ax_accuracy
    train_loss
    test_loss
    train_acc
    test_acc
end

function init_plot()
    if PLOT[1]
        fig = figure("Training history")
        clf()
        (ax_loss, ax_accuracy) = fig.subplots(2, 1, sharex=true)

        ax_loss.set_ylabel("Loss")
        ax_loss.autoscale_view()

        ax_accuracy.set_ylabel("Accuracy in %")
        # ax_accuracy.set_autoscaley_on(false)
        # ax_accuracy.set_autoscalex_on(true)
        ax_accuracy.set_ylim(0, 100)
        ax_accuracy.autoscale_view(scalex=true, scaley=false)
        xlabel("Epoch")

        train_loss = ax_loss.plot([], [], label="Train", color="orange")[1]
        test_loss = ax_loss.plot([], [], label="Test", color="green")[1]
        train_acc = ax_accuracy.plot([], [], color="orange")[1]
        test_acc = ax_accuracy.plot([], [], color="green")[1]

        ax_loss.legend()
        fig.tight_layout()
        PyPlot.show()
        global the_plot = Plot(fig, ax_loss, ax_accuracy, train_loss, test_loss, train_acc, test_acc)
    end
end

function remove_nothing(train_history)
    x = []
    y = []
    for i = 1:length(train_history)
        if train_history[i] !== nothing
            push!(x, x_history[i])
            push!(y, train_history[i])
        end
    end
    return x, y
end

function advance_history()
    if isempty(x_history)
        push!(x_history, 0)
    else
        push!(x_history, x_history[end] + 1)
    end

    if (HISTORY_TRAIN[1] && mod(x_history[end], HISTORY_TRAIN[2]) == 0) || (HISTORY_TEST[1] && mod(x_history[end], HISTORY_TEST[2]) == 0)
        println("Epoch $(x_history[end]):")
    end

    if HISTORY_TRAIN[1] && mod(x_history[end], HISTORY_TRAIN[2]) == 0
        train_loss, train_acc = loss_accuracy(:train)

        push!(train_loss_history, train_loss)
        push!(train_accuracy_history, train_acc)

        println("    train loss: ", train_loss_history[end])
        println("    train accurracy: $(train_accuracy_history[end])%")
        if PLOT[1] && mod(x_history[end], PLOT[2]) == 0
            x, y = remove_nothing(train_loss_history)
            the_plot.train_loss.set_data(x, y)
            x, y = remove_nothing(train_accuracy_history)
            the_plot.train_acc.set_data(x, y .* 100)
        end
    else
        push!(train_loss_history, nothing)
        push!(train_accuracy_history, nothing)
    end

    if HISTORY_TEST[1] && mod(x_history[end], HISTORY_TEST[2]) == 0
        test_loss, test_acc = loss_accuracy(:test)

        push!(test_loss_history, test_loss)
        push!(test_accuracy_history, test_acc)

        println("    test loss: ", test_loss_history[end])
        println("    test accurracy: $(test_accuracy_history[end])%")
        if PLOT[1] && mod(x_history[end], PLOT[2]) == 0
            x, y = remove_nothing(test_loss_history)
            the_plot.test_loss.set_data(x, y)
            x, y = remove_nothing(test_accuracy_history)
            the_plot.test_acc.set_data(x, y .* 100)
        end
    else
        push!(test_loss_history, nothing)
        push!(test_accuracy_history, nothing)
    end

    if PLOT[1]
        the_plot.ax_loss.relim()
        the_plot.ax_accuracy.relim()
        the_plot.ax_loss.autoscale_view()
        the_plot.ax_accuracy.autoscale_view(scalex=true, scaley=false)
        PyPlot.show()
    end
end

function init_model()
    if isempty(LOAD_PATH)
        global model = MODEL() |> device
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

function init_cuda()
    if USE_CUDA == true
        if CUDA.functional()
            global device = gpu
        else
            global device = cpu
            @warn "Despite USE_CUDA = true, CUDA is disabled as it isn't supported."
        end
    else
        global device = cpu
    end
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
X_traindata = nothing
Y_traindata = nothing
X_testdata = nothing
Y_testdata = nothing

# TRAINING
# *************************************************************************************************************************

loss(x, y) = LOSS(model(x), y)
opt = OPTIMIZER(LEARNING_RATE)

init_cuda()
init_plot()
init_model()

for epoch = 1:EPOCHS
    @showprogress "Epoch $(x_history[end]+1)..." for (x, y) in train_data
        x, y = x |> device, y |> device
        gs = Flux.gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end
    advance_history()
end

save_model()

#=
data_ = [Array{Float32}(data_[i,:,:]) for i in 1:size(data_)[1]]
data = Vector{Matrix{Float32}}(undef, length(data_))
for i = 1:length(data_)
    data[i] = reshape(data_[i], 1, :)
end
println("transformed data", length(data))
=#

end #module
module AI

using PyCall, Flux, PyPlot, BSON, ProgressMeter, CUDA
using CUDA: CuIterator
np = pyimport("numpy")
using Flux: crossentropy, train!, onecold


include("default_config.jl") # provide default options, don't change
include("config.jl") # overwrite default options, you just need to set the variables
include("common_functions.jl")

function get_data(path)
    return np.load(path)
end

function ignore_file(path)
    # Return true if file should be ignored
    name = split(path, "/")[end]
    if name[1] == "-"
        return true
    else
        return false
    end
end

function get_data_length(data_info)
    samples = 0
    for classification in data_info
        for file in readdir(classification[1])
            path = classification[1] * file
            if ignore_file(path) == false
                d = mapslices(rotr90, get_data(path), dims=[1, 3])
                samples += size(d)[3]
            end
        end
    end
    return samples
end

function set_all_data()
    i = [1, 0]
    for classification in TRAIN_DATA
        for file in readdir(classification[1])
            path = classification[1] * file
            if ignore_file(path) == false
                d = mapslices(rotr90, get_data(path), dims=[1, 3])
                i[2] += size(d)[3]
                X_traindata[:, 1, :, i[1]:i[2]] = d
                # println("i1: $(i[1]), i2: $(i[2])")
                # println(size(Y_traindata[:, i[1]:i[2]]))
                Y_traindata[:, i[1]:i[2]] .= classification[2]
                i[1] += size(d)[3]
            end
        end
    end

    i = [1, 0]
    for classification in TEST_DATA
        for file in readdir(classification[1])
            path = classification[1] * file
            if ignore_file(path) == false
                d = mapslices(rotr90, get_data(path), dims=[1, 3])
                i[2] += size(d)[3]
                X_testdata[:, 1, :, i[1]:i[2]] = d
                Y_testdata[:, i[1]:i[2]] .= classification[2]
                i[1] += size(d)[3]
            end
        end
    end
end

function init_data()
    global num_samples_train = get_data_length(TRAIN_DATA)
    global num_samples_test = get_data_length(TEST_DATA)

    # Pre-initialise arrays to improve performance
    global X_traindata = Array{Float32}(undef, MAX_FREQUENCY, 1, NUM_CHANNELS, num_samples_train)
    global Y_traindata = Array{Float32}(undef, num_outputs, num_samples_train)
    global X_testdata = Array{Float32}(undef, MAX_FREQUENCY, 1, NUM_CHANNELS, num_samples_test)
    global Y_testdata = Array{Float32}(undef, num_outputs, num_samples_test)

    set_all_data()
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
        println("    train accurracy: $(train_accuracy_history[end] * 100)%")
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
        println("    test accurracy: $(test_accuracy_history[end] * 100)%")
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
    if (!isempty(LOAD_PATH)) && isfile(LOAD_PATH)
        load_model()
        global model = model |> device
        println("Epoch $(x_history[end]):")
        if train_loss_history[end] !== nothing
            println("    train loss: ", train_loss_history[end])
        end
        if train_accuracy_history[end] !== nothing
            println("    train accurracy: $(train_accuracy_history[end] * 100)%")
        end
        if test_loss_history[end] !== nothing
            println("    test loss: ", test_loss_history[end])
        end
        if test_accuracy_history[end] !== nothing
            println("    test accurracy: $(test_accuracy_history[end] * 100)%")
        end
    else
        global model = MODEL() |> device
        global x_history = []
        global train_loss_history = []
        global test_loss_history = []
        global train_accuracy_history = []
        global test_accuracy_history = []
        advance_history()
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

init_data()

# Turn data into Flux DataLoaders
train_data = Flux.Data.DataLoader((X_traindata, Y_traindata), batchsize=BATCH_SIZE, shuffle=true, partial=false)
test_data = Flux.Data.DataLoader((X_testdata, Y_testdata), batchsize=BATCH_SIZE, shuffle=true, partial=false)

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
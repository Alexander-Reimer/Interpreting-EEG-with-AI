module AI

using PyCall, Flux, PyPlot, BSON, ProgressMeter, CUDA, Statistics
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

function clip_scale(data)
    data = clamp.(data, -10, 10) ./ 10
    return data
end

function set_all_data()
    i = [1, 0]
    for classification in TRAIN_DATA
        for file in readdir(classification[1])
            path = classification[1] * file
            if ignore_file(path) == false
                d = mapslices(rotr90, get_data(path), dims=[1, 3])
                d = clip_scale(d)
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
                d = clip_scale(d)
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
        global x_i = x
        global y_i = y
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
    for i = eachindex(train_history)
        if train_history[i] !== nothing
            push!(x, x_history[i])
            push!(y, train_history[i])
        end
    end
    return x, y
end

function advance_history()
    testmode!(model)
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
    trainmode!(model)
end

function init_model()
    if (!isempty(LOAD_PATH)) && isfile(LOAD_PATH)
        load_model()
        global model = model |> device
        testmode!(model)
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
        trainmode!(model)
    else
        global model = MODEL() |> device
        global x_history = []
        global train_loss_history = []
        global test_loss_history = []
        global train_accuracy_history = []
        global test_accuracy_history = []
        # advance_history()
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

function get_activations()
    global test_activations = []
    global train_activations = []

    last_activations = test_data.data[1]
    for i = 1:length(model)
        last_activations = model[i](last_activations)
        push!(test_activations, last_activations)
    end

    last_activations = train_data.data[1]
    for i = 1:length(model)
        last_activations = model[i](last_activations)
        push!(train_activations, last_activations)
    end

    global test_means = []
    global test_stds = []
    global train_means = []
    global train_stds = []
    for i3 = 1:length(train_activations)
        layer = i3
        if length(size(train_activations[layer])) == 4
            train_mean = zeros(Float32, size(train_activations[layer])[1], size(train_activations[layer])[3])
            # train_std = zeros(Float32, size(train_activations[layer])[1], size(train_activations[layer])[3])
            for i = 1:size(train_activations[layer])[1]
                for i2 = 1:size(train_activations[layer])[3]
                    train_mean[i, i2] = mean(train_activations[layer][i, 1, i2, :])
                    # train_std[i, i2] = std(train_activations[layer][i, 1, i2, :])
                end
            end

            test_mean = zeros(Float32, size(test_activations[layer])[1], size(test_activations[layer])[3])
            # test_std = zeros(Float32, size(test_activations[layer])[1], size(test_activations[layer])[3])
            for i = 1:size(test_activations[layer])[1]
                for i2 = 1:size(test_activations[layer])[3]
                    test_mean[i, i2] = mean(test_activations[layer][i, 1, i2, :])
                    # test_std[i, i2] = std(test_activations[layer][i, 1, i2, :])
                end
            end
        else
            train_mean = zeros(Float32, size(train_activations[layer])[1])
            # train_std = zeros(Float32, size(train_activations[layer])[1])
            for i = 1:size(train_activations[layer])[1]
                train_mean[i] = mean(train_activations[layer][i, :])
                # train_std[i] = std(train_activations[layer][i, :])
            end

            test_mean = zeros(Float32, size(test_activations[layer])[1])
            # test_std = zeros(Float32, size(test_activations[layer])[1])
            for i = 1:size(test_activations[layer])[1]
                test_mean[i] = mean(test_activations[layer][i, :])
                # test_std[i] = std(test_activations[layer][i, :])
            end
        end
        push!(test_means, test_mean)
        # push!(test_stds, test_std)
        push!(train_means, train_mean)
        # push!(train_stds, train_std)
    end
end

function remove(layer::Flux.Conv, i)
    c = i
    new_weights = cat(weights[:, :, :, 1:c-1],weights[:, :, :, c+1:end], dims=4)
    h, w, d, n = size(new_weights)
    new_l = Conv((h, w), d => n, relu)
    new_l.weight .= new_weights
    return new_l
end

function remove(layer::Flux.Dense, i)
    weights = layer.weight
    new_weights = cat(weights[1:i-1, :], weights[i+1:end, :], dims=1)
    h, w = size(new_weights)
    new_l = Dense(w, h, tanh)
    new_l.weight .= new_weights
    return new_l
end

function remove_next(layer::Flux.Dense, i)
    weights = layer.weight
    new_weights = cat(weights[:, 1:i-1], weights[:, i+1:end], dims=2)
    h, w = size(new_weights)
    new_l = Dense(w, h, tanh)
    new_l.weight .= new_weights
    return new_l
end

function remove(layer, i)
    return layer
end

function new_chain(layers)
    return Chain(layers...)
end

function adjust_network!()
    global model
    global layers
    diffs = train_means - test_means
    diffs = [abs.(i) for i in diffs]

    layers = [l for l in model.layers]
    global i2_2 = 0
    for i in w_layers
        i2 = argmax(diffs[i])[2]
        println(i2)
        l = layers[i]
        layers[i] = remove(l, i2)
        i2_2 = i2
    end
    layers[end] = remove_next(layers[end], i2_2)
    model = new_chain(layers)
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
# + sum(sqnorm, Flux.params(model)) # L2 weight regularisation

opt = OPTIMIZER(LEARNING_RATE)
NOISE ? noise(x) = NOISE_FUNCTION(x) : noise(x) = x

init_cuda()
init_plot()
init_model()

sqnorm(x) = sum(abs2, x)
loss(x, y) = LOSS(model(x), y) # + 0.1 * sum(sqnorm, Flux.params(model)) # L2 weight regularisation

@info "Gathering activations..."
get_activations()
@info "Adjusting network..."
adjust_network!()

#=
for epoch = 1:EPOCHS
    @showprogress "Epoch $(x_history[end]+1)..." for (x, y) in train_data
        x, y = noise(x |> device), y |> device
        gs = Flux.gradient(() -> loss(x, y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end
    advance_history()
    if train_accuracy_history[end] > test_accuracy_history[end] + 0.1
        get_activations()
        adjust_network!()
    end
end

# train_data = train_data |> device
save_model()
=#


end #module
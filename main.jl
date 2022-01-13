module AI

println("Loading BrainFlow...")
# For reading the EEG training samples
using BrainFlow
println("Loading Flux...")
# For creating and training the neural network
using Flux
println("Loading CUDA...")
# For using the (Nvidia) GPU if available for training
using CUDA
println("Loading PyPlot...")
# For plotting the costs
using PyPlot
println("Loading BSON...")
# For saving and loading the weights and costs
using BSON
using BSON: @load
println("Loading FFTW...")
# For the Discreete Fourier Transform
using FFTW

println("Loading Recover...")
# For loading corrupted endings, see recover_data.jl for more info
include("EEG.jl")
include("recover_data.jl")
println("Recover loaded!")

function make_fft(data)
    # Apply fft functions to data
    return get_magnitudes(split_double(fft(data)))
end

function split_double(fft_data)
    # Cut off the second half of the data because of mirror-effect
    half = round(Int, length(fft_data) / 2)
    fft_data = fft_data[1:half]
    # Double amplitudes to compensate
    fft_data .*= 2
    return fft_data
end

function get_magnitudes(fft_data)
    # Calculate the magnitude of the fft values (complex numbers)
    return abs.(fft_data)
end

function get_eeg_data(path, data_x, data_y, endings, output)
    sample_number = 1
    while isfile(path * string(sample_number) * ".csv")
        sample_data_x = BrainFlow.read_file(path * string(sample_number) * ".csv")[:, 1:2]

        for i = 1:size(sample_data_x)[2]
            BrainFlow.remove_environmental_noise(view(sample_data_x, :, i), 200, BrainFlow.FIFTY)
        end

        sample_data_x = reshape(sample_data_x, (:, 1))
        #sample_data_x[800] = endings[1][sample_number]
        #sample_data_x = [make_fft(sample_data_x[1:200])..., make_fft(sample_data_x[201:400])..., make_fft(sample_data_x[401:600])..., make_fft(sample_data_x[601:800])...]
        data_x = [data_x sample_data_x]
        sample_number += 1
    end

    for i = 1:sample_number-1
        data_y = [data_y output]
    end

    return data_x, data_y
end

function get_loader(train_portion = 0.9, blink_path = "Blink/", no_blink_path = "NoBlink/")
    # Load corrupted endings, see recover_data.jl for more info
    endings = Recover.get_endings()
    inputs_all_channels = 400
    outputs = 2
    data_x = Array{Float64}(undef, inputs_all_channels, 0)
    data_y = Array{Float64}(undef, outputs, 0)

    data_x, data_y = get_eeg_data(blink_path, data_x, data_y, endings, [1.0; 0.0])

    data_x, data_y = get_eeg_data(no_blink_path, data_x, data_y, endings, [0.0, 1.0])

    total_amount = round(Int, size(data_x)[2] / 2)
    amount_train = round(Int, total_amount * train_portion)

    # total amount of samples
    l = size(data_x)[2]
    # multiplying with train portion and dividing by two because it is used twice (one for test data, one for train data)
    l_train = round(Int, l * train_portion / 2)

    # dividing data into test and train data
    train_data_x = [data_x[:, 1:l_train] data_x[:, (l-l_train+1):l]]
    train_data_y = [data_y[:, 1:l_train] data_y[:, (l-l_train+1):l]]

    test_data_x = data_x[:, (l_train+1):l-l_train]
    test_data_y = data_y[:, (l_train+1):l-l_train]

    train_data = Flux.Data.DataLoader((train_data_x, train_data_y), batchsize = hyper_parameters.batch_size, shuffle = hyper_parameters.shuffle, partial = false)
    test_data = Flux.Data.DataLoader((test_data_x, test_data_y), batchsize = hyper_parameters.batch_size, shuffle = hyper_parameters.shuffle, partial = false)

    return train_data, test_data
end

function save_weights(model, name, test_losses, train_losses)
    model_weights = deepcopy(collect(params(model)))
    bson(name, Dict(:model_weights => model_weights, :test_losses => test_losses, :train_losses => train_losses))
end

function load_weights(name)
    content = BSON.load(name, @__MODULE__)
    return content[:model_weights], content[:test_losses], content[:train_losses]
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += Flux.Losses.mse(model(x), y) # agg
        acc += sum(Flux.onecold(cpu(model(x))) .== Flux.onecold(cpu(y)))
        num += size(x, 2)
    end
    return ls / num, acc / num
end

function confusion_matrix(data_loader, model)
    blink_count = 0
    blink_acc = 0
    no_blink_count = 0
    no_blink_acc = 0
    for (x, y) in data_loader
        model = model |> cpu
        est = model(x)
        if y[1, 1] == 1.0
            blink_count += 1
            if est[1, 1] > est[2, 1]
                blink_acc += 1
            end
        else
            no_blink_count += 1
            if est[1, 1] < est[2, 1]
                no_blink_acc += 1
            end
        end
    end
    blink_acc /= blink_count
    no_blink_acc /= no_blink_count

    # Real Blinks: [:, 1], Real No Blinks: [:, 2], Estimated Blinks: [1, :], Estimated No Blinks: [2, :]
    return [blink_acc (1-no_blink_acc); (1-blink_acc) no_blink_acc] ./ 2
end

function build_model()
    # Amount of inputs for all channels
    inputs = 400
    return Chain(
        Dense(inputs, round(Int, inputs / 2), σ),
        Dense(round(Int, inputs / 2), round(Int, inputs / 2), σ),
        #Dense(round(Int, inputs / 4), round(Int, inputs / 16), σ),
        Dense(round(Int, inputs / 2), 2, σ)
    )
end

function prepare_cuda()
    # Enable CUDA on GPU if functional
    if CUDA.functional()
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end
    return device
end

function plot_loss(epoch, frequency, test_data, model, device, train_data)
    if mod(epoch, frequency) == 0
        test_loss, test_acc = loss_and_accuracy(test_data, model, device)
        train_loss, train_acc = loss_and_accuracy(train_data, model, device)
    
        push!(test_losses, test_loss)
        push!(train_losses, train_loss)
        push!(test_accs, test_acc)
        push!(train_accs, train_acc)
    
        clf()
        plot(test_losses, color="blue")
        plot(test_accs, color="blue", linestyle="dashed")
        plot(train_losses, color="red")
        plot(train_accs, color="red", linestyle="dashed")
    
        println("$(epoch) Epochen von $(hyper_parameters.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")
    end
end

function new_network(test_data, train_data, model, device)
    @info "Creating new network"
    test_loss, test_acc = loss_and_accuracy(test_data, model |> device, device)
    train_loss, train_acc = loss_and_accuracy(train_data, model |> device, device)
    push!(test_losses, cpu(test_loss))
    push!(train_losses, cpu(train_loss))
end

function old_network()
    @info "Loading old network"
    model_weights, test_losses, train_losses = load_weights("model.bson")
    global test_losses = test_losses
    global train_losses = train_losses
    return model_weights
end

function train(new = false)
    figure("Blinzeln Trainieren")
    xlabel("Epochen in 100er Schritten")
    ylabel("")

    device = prepare_cuda()
    # Load the training data and create the model structure with randomized weights
    train_data, test_data = get_loader()
    model = build_model()
    global test_losses = Float64[]
    global train_losses = Float64[]
    global test_accs = Float64[]
    global train_accs = Float64[]

    # if new = true, create a new network
    if new
        new_network(test_data, train_data, model, cpu)
    else
        model_weights = old_network()
        Flux.loadparams!(model, model_weights)
    end

    # Move model to device (GPU or CPU)
    model = model |> device

    ps = Flux.params(model)
    opt = Descent(hyper_parameters.learning_rate)

    test_loss, test_acc = loss_and_accuracy(test_data, model, device)

    @info "Training"
    println("0 Epochen von $(hyper_parameters.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")

    plot(test_losses, "b")

    for epoch = 1:hyper_parameters.training_steps
        for (x, y) in train_data
            x, y = device(x), device(y) # transfer data to device
            gs = gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        plot_loss(epoch, 100, test_data, model, device, train_data)
    end

    # Move model back to CPU (if it already was, it just stays)
    cpu(model)
    model |> cpu
    test_loss, test_acc = loss_and_accuracy(test_data, model, device)
    println("Loss: $test_loss, Accuracy: $test_acc")

    save_weights(model, "model.bson", test_losses, train_losses)
    @info "Weights saved at \"model.bson\""
    println(confusion_matrix(test_data, model))
end

function test(model)
    counter = 200
    BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    params = BrainFlowInputParams(
        serial_port = "/dev/cu.usbmodem11"
    )
    board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
    samples = []
    #BrainFlow.release_session(board_shim)
    BrainFlow.prepare_session(board_shim)
    BrainFlow.start_stream(board_shim)
    sleep(1)
    println("Starting!")
    println("")
    for i = 1:100
        counter-= 20
        sample = EEG.get_some_board_data(board_shim, 200)
        clf()
        plot(sample)
        sample = reshape(sample, (:, 1))
        #sample = [make_fft(sample[1:200])..., make_fft(sample[201:400])..., make_fft(sample[401:600])..., make_fft(sample[601:800])...]
        y = model(sample)
        println(y[1], "    ", y[2])
        if y[1]>y[2] + 0.2
            counter += 100
            #println("hgizugz")
        end
        #push!(samples, sample)
        sleep(0.25)
        #print("\b\b\b\b\b")
        #println(counter)
    end
    BrainFlow.release_session(board_shim)

end

mutable struct Args
    learning_rate::Float64
    batch_size::Int
    training_steps::Int
    shuffle::Bool
    # cut off fft data (frequencies) below lower_limit or above upper_limit which also determines amount of inputs (inputs = upper_limit - lower_limit)
    lower_limit::Int
    upper_limit::Int
end

global hyper_parameters = Args(0.001, 2, 1000, true, 7, 13)

#train(true)


#device = prepare_cuda()
#model = build_model()
#parameters = old_network()
#Flux.loadparams!(model, parameters)
#test(model)


#=
model = build_model()
parameters = old_network()
Flux.loadparams!(model, parameters)

train_data, test_data = get_loader()

for i = 1:20
    sample = test_data.data[1][:,i]
    y = model(sample)
    if ((y[1] > y[2]) && (test_data.data[2][1, i] == 1.0)) || ((y[1] < y[2]) && (test_data.data[2][2, i] == 1.0))
        println("Correct!")
    else
        println("Incorrect!")
    end
end
=#
end # Module
module EEG

println("Loading BrainFlow...")
using BrainFlow
println("Loading Flux...")
using Flux
println("Loading CUDA...")
using CUDA
println("Loading PyPlot...")
using PyPlot
println("Loading BSON...")
using BSON
using BSON: @load
println("Loading FFTW...")
using FFTW
println("Packages loaded!")

println("Loading Recover...")
include("recover_data.jl")
println("Recover loaded!")

function make_fft(data)
    # Apply fft function to data
    return fft(data)
end

function split_double(fft_data)
    # Cut off the second half of the data because of mirror-effect
    half = round(Int, length(fft_data) / 2)
    fft_data = fft_data[1:half]
    # Double data to compensate
    fft_data .*= 2
    return fft_data
end

function get_magnitudes(fft_data)
    # Calculate the magnitude of the fft values (complex numbers)
    return abs.(fft_data)
end

function get_loader(train_portion = 0.9, blink_path="Blink/", no_blink_path="NoBlink/")
    endings = Recover.get_endings()
    train_data_x = Array{Float64}(undef, 400, 0)

    i = 1
    while isfile(blink_path * string(i) * ".csv")
        sample = BrainFlow.read_file(blink_path * string(i) * ".csv") |> transpose
        sample = reshape(sample, (:, 1))
        sample[800] = endings[1][i]
        sample_fft = Float64[]
        for i = 1:200:800
            fft_single_channel = make_fft(sample[i:(i+199)])
            fft_single_channel = split_double(fft_single_channel)
            fft_single_channel = get_magnitudes(fft_single_channel)
            append!(sample_fft, fft_single_channel)
        end
        train_data_x = [train_data_x sample_fft]
        i += 1
    end


    train_data_y = Array{Float64}(undef, 2, 0)
    for i = 1:size(train_data_x)[2]
        train_data_y = [train_data_y [1.0, 0.0]] # Output 1: Blink = 1, Output 2: No Blink = 0
    end

    i = 1
    while isfile(no_blink_path * string(i) * ".csv")
        sample = BrainFlow.read_file(blink_path * string(i) * ".csv") |> transpose
        sample = reshape(sample, (:, 1))
        sample[800] = endings[1][i]
        sample_fft = Float64[]
        for i = 1:200:800
            fft_single_channel = make_fft(sample[i:(i+199)])
            fft_single_channel = split_double(fft_single_channel)
            fft_single_channel = get_magnitudes(fft_single_channel)
            append!(sample_fft, fft_single_channel)
        end
        train_data_x = [train_data_x sample_fft]
        i += 1
    end

    for i = 1:(size(train_data_x)[2] - size(train_data_y)[2])
        train_data_y = [train_data_y [0.0, 1.0]] # Output 1: Blink = 0, Output 2: No Blink = 1
    end

    l = size(train_data_x)[2]
    l_train = round(Int, l * train_portion / 2)

    test_data_x = [train_data_x[:, 1:l_train] train_data_x[:, (l-l_train + 1) : l]]
    test_data_y = [train_data_y[:, 1:l_train] train_data_y[:, (l-l_train + 1) : l]]

    train_data_x = train_data_x[:, (l_train + 1) : l-l_train]
    train_data_y = train_data_y[:, (l_train + 1) : l-l_train]

    train_data = Flux.Data.DataLoader((train_data_x, train_data_y), batchsize=hyper_parameters.batch_size, shuffle=hyper_parameters.shuffle, partial=false)
    test_data = Flux.Data.DataLoader((test_data_x, test_data_y), batchsize=1, shuffle=hyper_parameters.shuffle, partial=false)
    
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
        num +=  size(x, 2)
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
            if est[1,1] > est[2,1]
                blink_acc += 1
            end
        else
            no_blink_count += 1
            if est[1,1] < est[2,1]
                no_blink_acc += 1
            end
        end
    end
    blink_acc /= blink_count
    no_blink_acc /= no_blink_count
    
    # Real Blinks: [:, 1], Real No Blinks: [:, 2], Estimated Blinks: [1, :], Estimated No Blinks: [2, :]
    return [blink_acc (1 - no_blink_acc); (1 - blink_acc) no_blink_acc]./2
end

function build_model()
    return Chain(
        Dense(400, 200, σ),
        Dense(200, 200, σ),
        Dense(200, 2, σ)
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
        clf()
        plot(test_losses, "b")
        plot(train_losses, "r")

        println("$(epoch) Epochen von $(hyper_parameters.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")
    end
end

function new_network(test_data, train_data, model, device)
    @info "Creating new network"
    test_loss, test_acc = loss_and_accuracy(test_data, model |> device, device)
    train_loss, train_acc = loss_and_accuracy(train_data, model |> device, device)
    push!(test_losses, test_loss)
    push!(train_losses, train_loss)
end

function old_network(train_data, model, device)
    @info "Loading old network"
    model_weights, test_losses, train_losses = load_weights("model.bson")
    global test_losses = test_losses
    global train_losses = train_losses
    return model_weights
end

function train(new = false)
    device = prepare_cuda()
    train_data, test_data = get_loader()
    # Load the training data and create the model structure with randomized weights
    model = build_model()
    global test_losses = Float64[]
    global train_losses = Float64[]
    if new
        new_network(test_data, train_data, model, device)
    else
        model_weights = old_network(test_data, model, device)
        Flux.loadparams!(model, model_weights)
    end

    model = model |> device

    ps = Flux.params(model)
    opt = Descent(hyper_parameters.learning_rate)

    test_loss, test_acc = loss_and_accuracy(test_data, model, device)

    println("0 Epochen von $(hyper_parameters.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")
    clf()
    plot(test_losses, "b")
    
    for epoch in 1:hyper_parameters.training_steps
        for (x, y) in train_data
            x, y = device(x), device(y) # transfer data to device
            gs = gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        plot_loss(epoch, 50, test_data, model, device, train_data)
    end

    cpu(model)
    model |> cpu
    test_loss, test_acc = loss_and_accuracy(test_data, model, device)
    println("Loss: $test_loss, Accuracy: $test_acc")

    save_weights(model, "model.bson", test_losses, train_losses)
    @info "Weights saved at \"model.bson\""
    println(confusion_matrix(test_data, model))
end


mutable struct Args
    learning_rate :: Float64
    batch_size :: Int
    training_steps :: Float64
    shuffle :: Bool
end

global hyper_parameters = Args(0.00001, 1, 1000, true)

train(true)
end # module

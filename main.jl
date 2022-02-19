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

function get_eeg_data(path, data_x, data_y, endings, output)
    sample_number = 1
    while isfile(path * string(sample_number) * ".csv")
        # Read recorded EEG data
        global sample_data_x = BrainFlow.read_file(path * string(sample_number) * ".csv")


        temp_data_x = Array{Float64}(undef, size(sample_data_x, 1), 0)

        for electrode in hyper_params.electrodes
            temp_data_x = [temp_data_x sample_data_x[:, electrode]]
        end
        sample_data_x = temp_data_x
        # Filter 50 Hz frequencies to remove environmental noise using BrainFlow
        for i = 1:size(sample_data_x)[2]
            if hyper_params.notch == 50
                notch = BrainFlow.FIFTY
            elseif hyper_params.notch == 60
                notch = BrainFlow.SIXTY
            else
                notch = hyper_params.notch
            end
            BrainFlow.remove_environmental_noise(view(sample_data_x, :, i), 200, notch)
        end

        sample_data_x = reshape(sample_data_x, (:, 1))
        if 4 in hyper_params.electrodes
            sample_data_x[last] = endings[1][sample_number]
        end

        if hyper_params.fft == true
            temp_data_x = []
            # Perform FFT on data, once per channel
            for i = 1:length(hyper_params.electrodes)
                s = (i - 1) * 200 + 1
                e = i * 200
                #println(size(sample_data_x[s:e]))
                fft_sample_x = abs.(rfft(sample_data_x[s:e]))
                fft_sample_x[hyper_params.notch + 1] = 0.0
                fft_sample_x = fft_sample_x[hyper_params.lower_limit:(hyper_params.upper_limit + 1)]
                append!(temp_data_x, fft_sample_x)
            end
            sample_data_x = temp_data_x
        end
        # Append to existing data
        data_x = [data_x sample_data_x]
        sample_number += 1
    end

    # Append given "output" to data_y ([1.0, 0.0] for Blink and [0.0, 1.0] for NoBlink)
    for i = 1:sample_number-1
        data_y = [data_y output]
    end

    return data_x, data_y
end

function get_loader(train_portion = 0.9, blink_path = "Blink/first_samples-before_01-15-2022/", no_blink_path = "NoBlink/first_samples-before_01-15-2022/")
    # Load corrupted endings, see recover_data.jl for more info
    # Not used right now as the 3. and 4. channel data isn't used at the moment
    endings = Recover.get_endings()

    inputs_all_channels = hyper_params.inputs
    outputs = 2 # amount of output neurons
    data_x = Array{Float64}(undef, inputs_all_channels, 0)
    data_y = Array{Float64}(undef, outputs, 0)

    data_x, data_y = get_eeg_data(blink_path, data_x, data_y, endings, [1.0; 0.0])

    data_x, data_y = get_eeg_data(no_blink_path, data_x, data_y, endings, [0.0, 1.0])

    # total amount of samples
    l = size(data_x)[2]

    # multiplying with train portion and dividing by two because it is used twice (one for test data, one for train data)
    l_train = round(Int, l * train_portion / 2)

    # dividing data into test and train data
    train_data_x = [data_x[:, 1:l_train] data_x[:, (l-l_train+1):l]]
    train_data_y = [data_y[:, 1:l_train] data_y[:, (l-l_train+1):l]]

    test_data_x = data_x[:, (l_train+1):l-l_train]
    test_data_y = data_y[:, (l_train+1):l-l_train]

    train_data = Flux.Data.DataLoader((train_data_x, train_data_y),
        batchsize = hyper_params.batch_size, shuffle = hyper_params.shuffle,
        partial = false
    )
    test_data = Flux.Data.DataLoader((test_data_x, test_data_y),
        batchsize = hyper_params.batch_size, shuffle = hyper_params.shuffle,
        partial = false)

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
    # Create a confusion matrix for 
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
    inputs = hyper_params.inputs
    return Chain(
        Dense(inputs, round(Int, inputs / 2), σ),
        Dense(round(Int, inputs / 2), round(Int, inputs / 2), σ),
        #Dense(round(Int, inputs / 4), round(Int, inputs / 16), σ),
        Dense(round(Int, inputs / 2), 2, σ)
    )
end

function prepare_cuda()
    # Enable CUDA on GPU if functional
    if hyper_params.cuda == true
        if CUDA.functional()
            @info "Training on CUDA GPU"
            CUDA.allowscalar(false)
            device = gpu
        else
            @info "No NVIDIA GPU detected --> Training on CPU"
            device = cpu
        end
    else
        @info "CUDA disabled --> Training on CPU"
        device = cpu
    end
    return device
end

function plot_loss(epoch, frequency, test_data, model, device, train_data; label=false)
    if mod(epoch, frequency) == 0
        if label
            test_loss_l = "Testdaten Cost"
            test_acc_l = "Testdaten Genauigkeit"
            train_loss_l = "Trainingsdaten Cost"
            train_acc_l = "Trainingsdaten Genauigkeit"
        else
            test_loss_l = ""
            test_acc_l = ""
            train_loss_l = ""
            train_acc_l = ""
        end
    
        test_loss, test_acc = loss_and_accuracy(test_data, model, device)
        train_loss, train_acc = loss_and_accuracy(train_data, model, device)
    
        push!(test_losses, test_loss)
        push!(train_losses, train_loss)
        push!(test_accs, test_acc * 100)
        push!(train_accs, train_acc * 100)
    
        #clf()
        ax1.plot(test_losses, color = "red", label=test_loss_l)
        ax1.plot(train_losses, color = "red", linestyle = "dashed", label=train_loss_l)

        ax2.plot(test_accs, color = "blue", label=test_acc_l)
        ax2.plot(train_accs, color = "blue", linestyle = "dashed", label=train_acc_l)
    
        println("$(epoch) Epochen von $(hyper_params.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")
    end
end

function new_network(test_data, train_data, model, device)
    @info "Creating new network"
    test_loss, test_acc = loss_and_accuracy(test_data, model |> device, device)
    train_loss, train_acc = loss_and_accuracy(train_data, model |> device, device)

    push!(test_losses, cpu(test_loss))
    push!(train_losses, cpu(train_loss))
    push!(test_accs, cpu(test_acc) * 100)
    push!(train_accs, cpu(train_acc) * 100)
end

function old_network()
    @info "Loading old network"
    model_weights, test_losses, train_losses = load_weights("model.bson")
    global test_losses = test_losses
    global train_losses = train_losses
    return model_weights
end

function train(new = false)
    global fig, ax1 = subplots()
    xlabel("Epochen in 100er Schritten")

    ylabel("Cost", color = "red")
    ax1.tick_params(axis = "y", color = "red", labelcolor = "red")

    global ax2 = twinx() # autoscalex_on=false
    ax2.set_ylim(ymin = 0, ymax = 100, auto = false)

    ylabel("Genauigkeit in %", color = "blue")
    ax2.tick_params(axis = "y", color = "blue", labelcolor = "blue")

    fig.tight_layout()
    global device = prepare_cuda()
    # Load the training data and create the model structure with randomized weights
    train_data, test_data = get_loader(0.9, "Blink/01-26-2022/", "NoBlink/01-26-2022/")
    #train_data, test_data = get_loader(0.9, "Blink/first_samples-before_01-15-2022/", "NoBlink/first_samples-before_01-15-2022/")
    
    global model = build_model()
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
    opt = Descent(hyper_params.learning_rate)

    test_loss, test_acc = loss_and_accuracy(test_data, model, device)
    train_loss, train_acc = loss_and_accuracy(train_data, model, device)

    @info "Training"
    println("0 Epochen von $(hyper_params.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")

    #plot(test_losses, "b")

    println(test_losses)
    println(test_accs)

    for epoch = 1:hyper_params.training_steps
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
    #println(confusion_matrix(test_data, model))

    plot_loss(100, 100, test_data, model, device, train_data, label = true)

    #fig.legend(loc = "center right")
    fig.tight_layout()
    #ax2.legend()
end

function test(model)
    counter = 200
    BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    params = BrainFlowInputParams(
        serial_port = "COM3"
    )
    board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
    samples = []
    if BrainFlow.is_prepared(board_shim)
        #BrainFlow.release_session(board_shim)
    end
    BrainFlow.prepare_session(board_shim)
    BrainFlow.start_stream(board_shim)
    sleep(1)
    println("Starting!")
    println("")
    blink_vals = []
    no_blink_vals = []
    for i = 1:100
        counter -= 20
        sample = EEG.get_some_board_data(board_shim, 200)
        clf()
        #plot(sample)
        sample = reshape(sample, (:, 1))
        sample = [make_fft(sample[1:200])..., make_fft(sample[201:400])...]
        y = model(sample)
        println(y[1], "    ", y[2], "       ", y[1] + y[2])
        push!(blink_vals, y[1])
        push!(no_blink_vals, y[2])
    
        #clf()
        plot(blink_vals, "green")
        plot(no_blink_vals, "red")

        #push!(samples, sample)
        sleep(0.25)
        #print("\b\b\b\b\b")
        #println(counter)
    end
    BrainFlow.release_session(board_shim)

end

mutable struct Args
    learning_rate::Float64
    training_steps::Int
    # cut off fft data (frequencies) below lower_limit or above upper_limit which also determines amount of inputs (inputs = upper_limit - lower_limit)
    lower_limit::Int
    upper_limit::Int
    electrodes::Array
    
    fft::Bool
    shuffle::Bool
    batch_size::Int
    notch::Int
    inputs::Int
    cuda::Bool
end

function Args(learning_rate, training_steps, lower_limit, upper_limit, electrodes; 
    fft = true, shuffle = true, batch_size = 2, notch = 50, cuda = true)
    
    inputs = (upper_limit - lower_limit + 2) * length(electrodes)
    return Args(learning_rate, training_steps, lower_limit, upper_limit, electrodes, fft, shuffle,
    batch_size, notch, inputs, cuda)
end

global hyper_params = Args(0.001, 1000, 1, 100, [1, 2, 3]; cuda = false)

train(true)


#=
device = prepare_cuda()
model = build_model()
parameters = old_network()
Flux.loadparams!(model, parameters)
#test(model)
=#

end # Module
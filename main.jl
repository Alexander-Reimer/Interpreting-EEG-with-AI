module AI
# 19:05:30 - 19:12:00
# 19:12:00 - 19:13:40
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

using WAV

println("Loading Recover...")
# For loading corrupted endings, see recover_data.jl for more info
include("recover_data.jl")
include("EEG.jl")
println("Recover loaded!")

function get_eeg_data(paths, data_x, data_y, output)
    for path in paths
        endings = Recover.get_endings_path(path)
        sample_number = 1
        while isfile(path * string(sample_number) * ".csv")
            # Read recorded EEG data
            global sample_data_x = BrainFlow.read_file(path * string(sample_number) * ".csv")

            temp_data_x = Array{Float64}(undef, size(sample_data_x, 1), 0)

            # (Try to) filter 50 / 60 Hz frequencies to remove environmental noise using BrainFlow
            # It doesnt really do a lot, but at least it's something
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

            # Make data 1d
            sample_data_x = reshape(sample_data_x, (:, 1))

            if 4 in hyper_params.electrodes
                sample_data_x[end] = endings[sample_number]
            end

            if hyper_params.fft == true
                temp_data_x = []
                # Perform FFT on all channels in hyper_params.electrodes
                for channel in hyper_params.electrodes
                    # Every channel has 200 values
                    s = (channel - 1) * 200 + 1
                    e = channel * 200

                    # Using rfft for better performance, as it is best for real values
                    fft_sample_x = abs.(rfft(sample_data_x[s:e]))
                    # Remove Amplitude for the frequency in hyper_params.notch
                    fft_sample_x[hyper_params.notch+1] = 0.0
                    # Cut off all frequencies not between hyper_params.lower_limit and hyper_params.upper_limit
                    fft_sample_x = fft_sample_x[hyper_params.lower_limit:(hyper_params.upper_limit+1)]

                    append!(temp_data_x, fft_sample_x)
                end
                sample_data_x = temp_data_x
            else
                temp_data_x = []
                # Only include data from the channels in hyper_params.electrodes
                for channel in hyper_params.electrodes
                    # Every channel has 200 values
                    s = (channel - 1) * 200 + 1
                    e = channel * 200
                    append!(temp_data_x, sample_data_x[s:e])
                end
                sample_data_x = temp_data_x
            end
            # Append to existing data
            data_x = [data_x sample_data_x]
            sample_number += 1
        end

        # Append given output to data_y ([1.0, 0.0] for Blink and [0.0, 1.0] for NoBlink)
        for i = 0:sample_number
            data_y = [data_y output]
        end
    end

    return data_x, data_y
end

function get_loader(train_portion=0.9, blink_paths=["Blink/first_samples-before_01-15-2022/"], no_blink_paths=["NoBlink/first_samples-before_01-15-2022/"])

    inputs_all_channels = hyper_params.inputs
    if hyper_params.one_out
        outputs = 1 # amount of output neurons
    else
        outputs = 2 # amount of output neurons
    end
    data_x = Array{Float64}(undef, inputs_all_channels, 0)
    data_y = Array{Float64}(undef, outputs, 0)

    if hyper_params.one_out
        blink_y = [1.0]
        no_blink_y = [0.0]
    else
        blink_y = [1.0; 0.0]
        no_blink_y = [0.0, 1.0]
    end

    data_x, data_y = get_eeg_data(blink_paths, data_x, data_y, blink_y)

    data_x, data_y = get_eeg_data(no_blink_paths, data_x, data_y, no_blink_y)

    # total amount of samples
    l = size(data_x)[2]

    # multiplying with train portion and dividing by two because it is used twice (one for test data, one for train data)
    l_train = round(Int, l * train_portion / 2)

    # Dividing data into test and train data by taking from the beginning and end for train, and then the middle for test
    # This way, the ratio of blink to no_blink data is equal in both
    train_data_x = [data_x[:, 1:l_train] data_x[:, (l-l_train+1):l]]
    train_data_y = [data_y[:, 1:l_train] data_y[:, (l-l_train+1):l]]
    test_data_x = data_x[:, (l_train+1):l-l_train]
    test_data_y = data_y[:, (l_train+1):l-l_train]

    train_data = Flux.Data.DataLoader((train_data_x, train_data_y),
        batchsize=hyper_params.batch_size, shuffle=hyper_params.shuffle,
        partial=false
    )
    test_data = Flux.Data.DataLoader((test_data_x, test_data_y),
        batchsize=hyper_params.batch_size, shuffle=hyper_params.shuffle,
        partial=false)

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

function loss_and_accuracy(data_loader, model, dev)
    loss = 0.0
    acc = 0.0
    num = 0
    model = dev(model)
    for (x, y) in data_loader
        x, y = dev(x), dev(y)
        ests = model(x)
        # Get differences between estimates of the model
        # and correct output
        diff = ests .- y
        # Calculate loss using the Mean Squared Error method
        loss += Flux.mean(diff .^ 2)
        # Round the values to either -1, 0, or 1
        # and then get the absolute values
        diff = round.(diff) .|> abs

        diff = 1 .- diff
        acc += sum(diff)
        num += size(y, 2)
    end
    return loss / num, acc / num / (Int(!hyper_params.one_out) + 1)
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
            if est[1, 1] > 1 - est[1, 1]
                blink_acc += 1
            end
        else
            no_blink_count += 1
            if est[1, 1] < 1 - est[1, 1]
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
    if hyper_params.one_out == true
        out_layer = Dense(round(Int, inputs / 4), 1, σ)
    else
        out_layer = Dense(round(Int, inputs / 4), 2, σ)
    end

    return Chain(
        Dense(inputs, round(Int, inputs / 4), σ),
        Dense(round(Int, inputs / 4), round(Int, inputs / 4), σ),
        #Dense(round(Int, inputs / 4), round(Int, inputs / 16), σ),
        out_layer
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

function plot_loss(epoch, frequency, test_data, model, device, train_data; label=false, plot=true)
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

        x = [i for i = 0:frequency:epoch]

        test_loss, test_acc = loss_and_accuracy(test_data, model, device)
        train_loss, train_acc = loss_and_accuracy(train_data, model, device)

        push!(test_losses, test_loss)
        push!(train_losses, train_loss)
        push!(test_accs, test_acc * 100)
        push!(train_accs, train_acc * 100)

        println("$(epoch) Epochen von $(hyper_params.training_steps): Loss ist $test_loss, Accuracy ist $test_acc.")
        #clf()
        if plot

            ax1.plot(test_losses, color="red", label=test_loss_l)
            ax1.plot(train_losses, color="red", linestyle="dashed", label=train_loss_l)

            ax2.plot(test_accs, color="blue", label=test_acc_l)
            ax2.plot(train_accs, color="blue", linestyle="dashed", label=train_acc_l)
        end
    end
end

function new_network(test_data, train_data, model, device)
    @info "Creating new network"
    test_loss, test_acc = loss_and_accuracy(test_data, model, cpu)
    train_loss, train_acc = loss_and_accuracy(train_data, model, cpu)

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

function train(subplot, plot_title, new=false, legend_entries=false)

    global ax1 = plt.subplot(subplot, title=plot_title)
    xlabel("Epochen in 100er Schritten")

    ylabel("Cost", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")

    global ax2 = twinx() # autoscalex_on=false
    ax2.set_ylim(ymin=0, ymax=100, auto=false)

    ylabel("Genauigkeit in %", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")

    #line_loss_test = ax1.plot(test_losses, color = "red")
    #line_loss_train = ax1.plot(train_losses, color = "red", linestyle = "dashed")

    #line_acc_test = ax2.plot(test_accs, color = "blue")
    #line_acc_train = ax2.plot(train_accs, color = "blue", linestyle = "dashed")

    fig.tight_layout()
    global device = prepare_cuda()
    # Load the training data and create the model structure with randomized weights
    global train_data, test_data = get_loader(0.9, ["Blink/Okzipital-03-16-2022/"], ["NoBlink/Okzipital-03-16-2022/"])
    #global train_data, test_data = get_loader(0.9, ["Blink/first_samples-before_01-15-2022/"], ["NoBlink/first_samples-before_01-15-2022/"])

    global model = device(build_model())
    global test_losses = Float64[]
    global train_losses = Float64[]
    global test_accs = Float64[]
    global train_accs = Float64[]

    # if new = true, create a new network
    if new
        new_network(test_data, train_data, model, device)
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
        plot_loss(epoch, hyper_params.plot_frequency, test_data, model, device, train_data, plot=false)
    end

    # Move model back to CPU (if it already was, it just stays)
    cpu(model)
    model |> cpu
    test_loss, test_acc = loss_and_accuracy(test_data, model, device)
    println("Loss: $test_loss, Accuracy: $test_acc")

    save_weights(model, "model.bson", test_losses, train_losses)
    @info "Weights saved at \"model.bson\""
    println(confusion_matrix(test_data, model))

    plot_loss(hyper_params.plot_frequency, hyper_params.plot_frequency, test_data, model, device, train_data, label=legend_entries)
    #ax2.legend()
end

function setup_robot()
    ev3dev_path = "../ev3dev.jl/ev3dev.jl"
    include(ev3dev_path)
end

function test(model)
    setup("Z:/Programming/EEG/mount/sys/class/")

    left_motor = Motor(:outB)
    right_motor = Motor(:outD)

    robot = Robot(left_motor, right_motor)

    drive(robot, 0)

    counter = 200
    BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    params = BrainFlowInputParams(
        serial_port="COM3"
    )
    board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
    samples = []
    if BrainFlow.is_prepared(board_shim)
        BrainFlow.release_session(board_shim)
    end
    BrainFlow.prepare_session(board_shim)
    BrainFlow.start_stream(board_shim)
    sleep(1)
    println("Starting!")
    println("")
    blink_vals = [0.0]
    no_blink_vals = [0.0]
    x = [0.0]

    if hyper_params.one_out
        global fig = figure("Live-Test mit einem Output-Neuron")
    else
        global fig = figure("Live-Test mit zwei Output-Neuronen")
    end

    clf()

    ylabel("Sicherheit des NN in %")
    xlabel("Zeit")
    fig.axes[1].set_ylim(0:1)

    line1 = plot([0], [0], "green")
    #line2 = plot([0], [0], "red")
    for i = 1:200
        for i = 1:500
            counter -= 20
            sample = EEG.get_some_board_data(board_shim, 200)
            #clf()
            #plot(sample)
            for i = 0:3
                BrainFlow.remove_environmental_noise(sample[i*200+1:(i+1)*200], 200, BrainFlow.FIFTY)
            end
            sample = reshape(sample, (:, 1))
            sample = [abs.(rfft(sample[1:200]))..., abs.(rfft((sample[201:400])))...] #, abs.(rfft((sample[401:600])))..., abs.(rfft((sample[601:800])))...
            y = model(sample)
            println(y)
            push!(blink_vals, y[1])
            #push!(no_blink_vals, y[2])

            #push!(x, x[end] + 0.01)

            #clf()
            #plot(blink_vals, "green")
            #line2[1].set_data(x, no_blink_vals)

            if y[1] > 0.7
                #=
                fs = 8e3
                t = 0.0:1/fs:prevfloat(0.01)
                f = 500
                y = sin.(2pi * f * t) * 0.1
                wavplay(y, fs)
                =#
                drive(robot, 50)
            else
                drive(robot, 0)
                #sleep(0.01)
            end

            #plot(no_blink_vals, "red")

            #push!(samples, sample)
            #print("\b\b\b\b\b")
            #println(counter)
        end
        line1[1].remove()
        line1 = plot(blink_vals, "green")
        #line1[1].set_data(x, blink_vals)
    end
    BrainFlow.release_session(board_shim)
    drive(robot, 0)
    drive(robot, 0)
    drive(robot, 0)
    #clf()
    #plot(blink_vals, "green", label = "Augen zu")
    #plot(blink_vals, "red", label = "Augen auf")
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
    one_out::Bool
    plot_frequency::Int
end

function Args(learning_rate, training_steps, lower_limit, upper_limit, electrodes;
    plot_frequency=200, fft=true, shuffle=true, batch_size=2, notch=50, cuda=true, one_out=false)

    if fft
        inputs = (upper_limit - lower_limit + 2) * length(electrodes)
    else
        inputs = length(electrodes) * 200
    end
    return Args(learning_rate, training_steps, lower_limit, upper_limit, electrodes, fft, shuffle,
        batch_size, notch, inputs, cuda, one_out, plot_frequency)
end

global hyper_params = Args(0.001, 1000, 1, 100, [1, 2, 3, 4]; cuda=false, one_out=true, plot_frequency=50, fft=false)

fig = figure("Trainingsprozess EEG: FFT vs. kein FFT")
train(312, "Ohne FFT", true)
global hyper_params = Args(0.001, 1000, 1, 100, [1, 2, 3, 4]; cuda=false, one_out=true, plot_frequency=50, fft=true)
train(313, "Mit FFT", true, true)
#=
sfig = plt.subplot(311)
test_loss_l = "Testdaten Cost"
test_acc_l = "Testdaten Genauigkeit"
train_loss_l = "Trainingsdaten Cost"
train_acc_l = "Trainingsdaten Genauigkeit"

sfig.plot([], color="red", label=test_loss_l)
sfig.plot([], color="red", linestyle="dashed", label=train_loss_l)

sfig.plot([], color="blue", label=test_acc_l)
sfig.plot([], color="blue", linestyle="dashed", label=train_acc_l)
=#
fig.legend(loc="upper left")
fig.tight_layout()

#=
setup_robot()
device = prepare_cuda()
model = build_model()
parameters = old_network()
Flux.loadparams!(model, parameters)
test(model)
# =#

end # Module
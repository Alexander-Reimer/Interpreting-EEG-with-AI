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
using DelimitedFiles
using WAV

using Interpolations

println("Loading Recover...")
# For loading corrupted endings, see recover_data.jl for more info
include("recover_data.jl")
include("EEG.jl")
println("Recover loaded!")

function get_eeg_data(paths, data_x, data_y, output)
    for path in paths
        #endings = Recover.get_endings_path(path)
        sample_number = 1
        while isfile(path * string(sample_number) * ".csv")
            # Read recorded EEG data
            global file_data_x = BrainFlow.read_file(path * string(sample_number) * ".csv")

            if typeof(file_data_x) != Array{Float64,2}
                #file_data_x = transpose(file_data_x)
            end

            if size(file_data_x)[2] !== 4
                file_data_x = file_data_x[:, 2:5]
            end

            additional_steps = size(file_data_x)[1] - 199

            for i = 1:additional_steps
                # temp_data_x = Array{Float64}(undef, size(sample_data_x, 1), 0)

                sample_data_x = file_data_x[0+i:199+i, :]
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
                    #sample_data_x[end] = endings[sample_number]
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
                # Append given output to data_y ([1.0, 0.0] for Blink and [0.0, 1.0] for NoBlink)
                data_y = [data_y output]
            end
            sample_number += 1
        end
    end
    return data_x, data_y
end

function get_loader(blink_paths, no_blink_paths, train_portion=0.9)

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

function save_weights(model, name, test_losses, train_losses, test_accs, train_accs, epoch)
    model_weights = deepcopy(collect(Flux.params(model)))
    bson(name, Dict(
        :model_weights => model_weights,
        :test_losses => test_losses,
        :train_losses => train_losses,
        :test_accs => test_accs,
        :train_accs => train_accs,
        :epoch => epoch
    ))
end

function load_data(name)
    content = BSON.load(name, @__MODULE__)
    return content
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

function confusion_matrix(data_loader, model, threshold=0.6)
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
            if est[1, 1] > threshold
                blink_acc += 1
            end
        else
            no_blink_count += 1
            if est[1, 1] < threshold
                no_blink_acc += 1
            end
        end
    end
    blink_acc /= blink_count
    no_blink_acc /= no_blink_count

    # Real Blinks: [:, 1], Real No Blinks: [:, 2], Estimated Blinks: [1, :], Estimated No Blinks: [2, :]
    return [blink_acc (1-blink_acc); (1-no_blink_acc) no_blink_acc] ./ 2
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
            #CUDA.allowscalar(false)
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

function plot_loss(training_plot, epoch, last_epoch, test_data, train_data, model, device; plot=true)
    x = [i for i = 0:hyper_params.plot_frequency:epoch]

    test_loss, test_acc = loss_and_accuracy(test_data, model, device)
    train_loss, train_acc = loss_and_accuracy(train_data, model, device)

    push!(test_losses, test_loss)
    push!(train_losses, train_loss)
    push!(test_accs, test_acc * 100)
    push!(train_accs, train_acc * 100)

    println("$(epoch) epochs of $(last_epoch + hyper_params.training_steps): loss is $test_loss, accuracy is $test_acc.")

    if plot
        training_plot.line_loss_test.set_data(x, test_losses)
        training_plot.line_loss_train.set_data(x, train_losses)

        training_plot.line_acc_test.set_data(x, test_accs)
        training_plot.line_acc_train.set_data(x, train_accs)

        training_plot.axe_cost.relim()
        training_plot.axe_cost.autoscale_view()

        training_plot.axe_acc.relim()
        training_plot.axe_acc.autoscale_view()
    end
end

function load_network!(path)
    @info "Loading old network"
    data = load_data(path)
    Flux.loadparams!(model, data[:model_weights])

    global test_losses = data[:test_losses]
    global train_losses = data[:train_losses]
    global test_accs = data[:test_accs]
    global train_accs = data[:train_accs]
    return data[:epoch]
end

mutable struct TrainingPlot
    axe_cost
    axe_acc
    line_loss_test
    line_loss_train
    line_acc_test
    line_acc_train
    legend_entries::Bool
end

function create_training_plot(plot_title; legend_entries=true, subplot=111)
    # Plotting stuff
    global axe_cost = plt.subplot(subplot, title=plot_title)
    xlabel("Epochen in $(hyper_params.plot_frequency)er Schritten")

    ylabel("Cost", color="red")
    axe_cost.tick_params(axis="y", color="red", labelcolor="red")

    global axe_acc = twinx() # autoscalex_on=false
    axe_acc.set_ylim(ymin=0, ymax=100, auto=false)

    ylabel("Genauigkeit in %", color="blue")
    axe_acc.tick_params(axis="y", color="blue", labelcolor="blue")

    if legend_entries
        test_loss_l = "Testdaten Cost"
        train_loss_l = "Trainingsdaten Cost"
        test_acc_l = "Testdaten Genauigkeit"
        train_acc_l = "Trainingsdaten Genauigkeit"
    else
        test_loss_l = ""
        train_loss_l = ""
        test_acc_l = ""
        train_acc_l = ""
    end

    line_loss_test = axe_cost.plot([], color="red", label=test_loss_l)[1]
    line_loss_train = axe_cost.plot([], color="red", linestyle="dashed", label=train_loss_l)[1]

    line_acc_test = axe_acc.plot([], color="blue", label=test_acc_l)[1]
    line_acc_train = axe_acc.plot([], color="blue", linestyle="dashed", label=train_acc_l)[1]

    training_plot = TrainingPlot(
        axe_cost,
        axe_acc,
        line_loss_test,
        line_loss_train,
        line_acc_test,
        line_acc_train,
        legend_entries
    )
    return training_plot
end

function train(training_plot, new=false)
    # what device should be used (CPU or GPU)
    global device = prepare_cuda()

    # Load the training data
    # EEG Okzipital:
    global train_data, test_data = get_loader(
        ["Blink/Okzipital-03-16-2022/"], #, "Blink/Okzipital-03-13-2022/", "Blink/Okzipital-03-13-2022-2/"],
        ["NoBlink/Okzipital-03-16-2022/"], 0.9) #, "NoBlink/Okzipital-03-13-2022/", "NoBlink/Okzipital-03-13-2022-2/"], 0.9)
    # EMG above eyes:
    #global train_data, test_data = get_loader(0.9, ["Blink/first_samples-before_01-15-2022/"], ["NoBlink/first_samples-before_01-15-2022/"])

    # create the model structure with randomized weights
    global model = build_model()

    if new
        # keep random weights
        @info "Creating new network"
        global test_losses = Float64[]
        global train_losses = Float64[]
        global test_accs = Float64[]
        global train_accs = Float64[]

        last_epoch = 0
        plot_loss(training_plot, 0, last_epoch, test_data, train_data, model, device)
    else
        # load weights from model.bson and get past accs, losses, & number of epochs
        last_epoch = load_network!(hyper_params.save_path)
    end

    # Move model to device (GPU or CPU)
    model = model |> device

    ps = Flux.params(model)
    # Use gradient descent as optimizer
    opt = Descent(hyper_params.learning_rate)

    @info "Training"

    for epoch = (last_epoch+1):(hyper_params.training_steps+last_epoch)
        for (x, y) in train_data
            x, y = device(x), device(y) # transfer data to device
            gs = Flux.gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        if mod(epoch, hyper_params.plot_frequency) == 0
            plot_loss(training_plot, epoch, last_epoch, test_data, train_data, model, device)
        end
    end

    # Move model back to CPU (if it already was, it just stays)
    cpu(model)
    model |> cpu
    # Plot & print loss if the last epoch wasnt multiple of frequency and thus not plotted / printed
    if mod(hyper_params.training_steps + last_epoch, hyper_params.plot_frequency) != 0
        plot_loss(training_plot, epoch, last_epoch, test_data, train_data, model, device)
    end

    save_weights(model, hyper_params.save_path, test_losses, train_losses, test_accs, train_accs, hyper_params.training_steps + last_epoch)
    @info "Weights saved at \"$(hyper_params.save_path)\""

    println(confusion_matrix(test_data, model))
end

function setup_robot()
    ev3dev_path = "../ev3dev.jl/ev3dev.jl"
    include(ev3dev_path)
end

function test(model, drive_robot)
    if drive_robot
        # Rework & Clean up!!!
        setup("Z:/Programming/EEG/mount/sys/class/")

        left_motor = Motor(:outB)
        right_motor = Motor(:outD)

        robot = Robot(left_motor, right_motor)

        drive(robot, 0)
    end

    counter = 200
    BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    params = BrainFlowInputParams(
        serial_port="COM4"
    )
    global board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
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

    if !drive_robot
        # define variables for playing the tone
        fs = 8e3
        t = 0.0:1/fs:prevfloat(0.01)
        f = 500
        y = sin.(2pi * f * t) * 0.1
    end

    for i = 1:40
        for i = 1:10
            sleep(0.01)
            sample = EEG.get_some_board_data(board_shim, 200)
            println("EEG #1: ", )
            #clf()
            #plot(sample)
            for i = 0:3
                BrainFlow.remove_environmental_noise(sample[i*200+1:(i+1)*200], 200, BrainFlow.FIFTY)
            end
            sample = reshape(sample, (:, 1))
            println("EEG: ", sample[10])
            sample = [abs.(rfft(sample[1:200]))[1:21]..., abs.(rfft((sample[201:400])))[1:21]..., abs.(rfft((sample[401:600])))[1:21]..., abs.(rfft((sample[601:800])))[1:21]...]
            println("FFT: ", sample[10])
            y = model(sample)
            println(y)
            push!(blink_vals, y[1])
            #push!(no_blink_vals, y[2])
        
            #push!(x, x[end] + 0.01)
        
            #clf()
            #plot(blink_vals, "green")
            #line2[1].set_data(x, no_blink_vals)
        
            if y[1] > 0.7
                if drive_robot
                    drive(robot, 50)
                else
                    print("\a")
                    #wavplay(y, fs)
                end
            else
                if drive_robot
                    drive(robot, 0)
                end
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
    if drive_robot
        drive(robot, 0)
        drive(robot, 0)
        drive(robot, 0)
    end
    #clf()
    #plot(blink_vals, "green", label = "Augen zu")
    #plot(blink_vals, "red", label = "Augen auf")
end

function get_data(path)
    io = open(path)
    content = readlines(io)
    close(io)
    outputs = []
    for i = 1:200
        output = Array{Float64}(undef, length(content), 4)
        for i2 = i:i+199
            #println("i2: ",i2)
            bar = split(content[i2], "\t")[2:5]
            foo = []
            for i3 = 1:4
                push!(foo, parse(Float64, String(bar[i3])))
            end
            output[i2, 1:4] = foo
        end
        #println(output)
        push!(outputs, output)
    end

    return outputs
end

function get_data_foo(path)

end

function save_predictions(data, model, path)
    blink_predictions = Float64[]
    noblink_predictions = Float64[]

    for (x, y) in data
        est = model(x)
        if y[1] == 1.0
            push!(blink_predictions, est[1])
        elseif y[1] == 0.0
            push!(noblink_predictions, est[1])
        else
            println(y)
            @error "Whatjshnsou?"
        end
    end

    io = open(path * "predictions-blink.txt", "w") do io
        writedlm(io, blink_predictions)
    end

    io = open(path * "predictions-noblink.txt", "w") do io
        writedlm(io, noblink_predictions)
    end

    return blink_predictions, noblink_predictions
end

function test2(path, model, color)

    global predicts = []
    for i2 = 1:24
        data2 = get_data(path * string(i2) * ".csv")
        for i = 1:200
            data = data2[i]
            data = reshape(data, (:, 1))
            temp_data_x = []
            # Perform FFT on all channels in hyper_params.electrodes
            for channel in 1:4
                # Every channel has 200 values
                s = (channel - 1) * 200 + 1
                e = channel * 200

                # Using rfft for better performance, as it is best for real values
                fft_sample_x = abs.(rfft(data[s:e]))
                # Remove Amplitude for the frequency in hyper_params.notch
                fft_sample_x[hyper_params.notch+1] = 0.0
                # Cut off all frequencies not between hyper_params.lower_limit and hyper_params.upper_limit
                fft_sample_x = fft_sample_x[hyper_params.lower_limit:(hyper_params.upper_limit+1)]

                append!(temp_data_x, fft_sample_x)
            end
            sample = copy(temp_data_x)
            push!(predicts, model(sample)[1])
        end
        println(i2)
    end

    io = open(path * "predictions.txt", "w") do io
        writedlm(io, predicts)
    end

    #hist(predicts, color=color)
    #println(predicts)
    return predicts
end



function test3(path, label; color="blue", fig="")
    figure(fig)
    io = open(path, "r") do io
        global content = readlines(io)
    end
    global predictions = parse.(Float64, content)
    figure("Histogramm von geschlossenen Augen")
    #clf()
    xlabel("Ausgabe des Neuronalen Netzwerks")
    ylabel("Häufigkeit der Ausgabe")
    #yscale(0.25)
    #xscale("log")

    bins = [0.0:0.01:1.0...]
    global a = hist(predictions, bins=bins, label=label, color=color, histtype="step", linewidth=1.2)
    itp_a = interpolate(a[1], BSpline(Cubic(Reflect(OnCell()))))

    plot(bins[2:end], itp_a, color=color, linestyle="--", linewidth=0.9)
    #indices = findall(iszero, predictions)
    #for i = length(indices):-1:1
    #  deleteat!(predictions, indices[i])
    #end
end

mutable struct Args
    learning_rate::Float64
    training_steps::Int
    # cut off fft data (frequencies) below lower_limit or above upper_limit which 
    # also determines amount of inputs (inputs = upper_limit - lower_limit)
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
    save_path::String
end

function Args(learning_rate, training_steps, save_path; lower_limit=1, upper_limit=100, electrodes=[1, 2, 3, 4],
    plot_frequency=200, fft=true, shuffle=true, batch_size=2, notch=50, cuda=true, one_out=false)
    if fft
        inputs = (upper_limit - lower_limit + 2) * length(electrodes)
    else
        inputs = length(electrodes) * 200
    end
    return Args(learning_rate, training_steps, lower_limit, upper_limit, electrodes, fft, shuffle,
        batch_size, notch, inputs, cuda, one_out, plot_frequency, save_path)
end

# global hyper_params = Args(0.001, 1000, "model.bson", cuda=false, one_out=true, plot_frequency=100, fft=false, batch_size=1)

#global hyper_params = Args(0.001, 1000, "model.bson", cuda=false, one_out=true, plot_frequency=100, fft=true, lower_limit=1, upper_limit=20, batch_size = 2)

#=
fig = figure("Training plot #1")
training_plot = create_training_plot("")
train(training_plot, true)


#=
fig = figure("Trainingsprozess EEG: FFT vs. kein FFT")

global hyper_params = Args(0.001, 50, "model.bson", cuda=false, one_out=true, plot_frequency=10, fft=false)
no_fft_plot = create_training_plot("Kein FFT", subplot=312, legend_entries = false)
train(no_fft_plot, true)

global hyper_params = Args(0.001, 50, "model.bson", cuda=false, one_out=true, plot_frequency=10, fft=true)
fft_plot = create_training_plot("FFT", subplot=313)
train(fft_plot, true)

fig.legend(loc="upper left")
fig.tight_layout()
=#

#=
global hyper_params = Args(0.001, 50, "model.bson", cuda=false, one_out=true, plot_frequency=10, fft=true)
# setup_robot()
device = prepare_cuda()
global model = build_model()

last_epoch = load_network!(hyper_params.save_path)

global train_data, test_data = get_loader(0.9, ["Blink/Okzipital-03-16-2022/"], ["NoBlink/Okzipital-03-16-2022/"])

#test(model, false)
# =# =#


global hyper_params = Args(0.001, 0, "model.bson", cuda=false, one_out=true, plot_frequency=100, fft=true, lower_limit=1, upper_limit=20, batch_size=1)
global model = build_model()
load_network!("model.bson")
# test(model, false)


#=
global blink_pred = test2("Blink/livetest_data/Okzipital-05-18-2022/", model, "green")
global noblink_pred = test2("NoBlink/livetest_data/Okzipital-05-18-2022/", model, "red")


# global train_data, test_data = get_loader(
#     ["Blink/Okzipital-03-16-2022/", "Blink/Okzipital-03-13-2022/", "Blink/Okzipital-03-13-2022-2/"],
#     ["NoBlink/Okzipital-03-16-2022/", "NoBlink/Okzipital-03-13-2022/", "NoBlink/Okzipital-03-13-2022-2/"], 0.9)


#save_predictions(test_data, model, "")

test3("NoBlink/livetest_data/Okzipital-05-18-2022/predictions.txt", "geöffnete Augen", color="red", fig="fig1")
test3("Blink/livetest_data/Okzipital-05-18-2022/predictions.txt", "geschlossene Augen", color="blue", fig="fig1")

# test3("predictions-noblink.txt", "geöffnete Augen", color="red", fig="fig3")
# test3("predictions-blink.txt", "geschlossene Augen", color="blue", fig="fig3")
legend()
# =#
end # Module
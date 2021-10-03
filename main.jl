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
println("Packages loaded!")

println("Loading Recover...")
include("recover_data.jl")
println("Recover loaded!")

function clean_nans(loader)
    for i = 1:size(loader.data[1])[2]
        
        if isnan(loader.data[1][800, i])
            println("NaN at i!")
            loader.data[1][800, i] = 0.0
        end

    end
    return loader
end

function get_loader(blink_path="Blink/", no_blink_path="NoBlink/")
    endings = Recover.get_endings()
    train_data_x = Array{Float64}(undef, 800, 0)
    i = 1
    while isfile(blink_path * string(i) * ".csv")
        sample = BrainFlow.read_file(blink_path * string(i) * ".csv") |> transpose
        sample = reshape(sample, (:, 1))
        train_data_x = [train_data_x sample]
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
        train_data_x = [train_data_x sample]
        i += 1
    end

    for i = 1:(size(train_data_x)[2] - size(train_data_y)[2])
        train_data_y = [train_data_y [0.0, 1.0]] # Output 1: Blink = 0, Output 2: No Blink = 1
    end

    both_ends = [endings[1]..., endings[2]...]
    for i = 1:length(both_ends)
        train_data_x[800, i] = both_ends[i]
    end

    train_data = Flux.Data.DataLoader((train_data_x, train_data_y), batchsize=hyper_parameters.batch_size, shuffle=hyper_parameters.shuffle, partial=false)
    
    return clean_nans(train_data)
end

function print_loader(loader)
    for epoch in 1:200
         for (x, y) in loader  # access via tuple destructuring
            println("Inputs:")
            println(IOContext(stdout, :compact => true, :limit => true), x)
            println(length(x))
            println("Outputs:")
            println(IOContext(stdout, :compact => true, :limit => true), y)
         end
    end
end

function save_weights(model, name, losses)
    model_weights = deepcopy(collect(params(model)))
    bson(name, Dict(:model_weights => model_weights, :losses => losses))
end

function load_weights(name)
    content = BSON.load(name, @__MODULE__)
    return content[:model_weights], content[:losses]
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
        Dense(800, 400, σ),
        Dense(400, 200, σ),
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

function plot_loss(epoch, frequency, train_data, model, device)
    if mod(epoch, frequency) == 0
        train_loss, train_acc = loss_and_accuracy(train_data, model, device)
        push!(train_losses, train_loss)
        clf()
        plot(train_losses, "b")
        println("$(epoch) Epochen von $(hyper_parameters.training_steps): Loss ist $train_loss, Accuracy ist $train_acc.")
    end
end

function new_network(train_data, model, device)
    @info "Creating new network"
    train_loss, train_acc = loss_and_accuracy(train_data, model |> device, device)
    push!(train_losses, train_loss)
end

function old_network(train_data, model, device)
    @info "Loading old network"
    model_weights, train_losses = load_weights("model.bson")
    global train_losses = train_losses
    return model_weights
end

function train(new = false)
    device = prepare_cuda()
    train_data = get_loader()
    # Load the training data and create the model structure with randomized weights
    model = build_model()
    global train_losses = Float64[]
    if new
        new_network(train_data, model, device)
    else
        model_weights = old_network(train_data, model, device)
        Flux.loadparams!(model, model_weights)
    end

    model = model |> device

    ps = Flux.params(model)
    opt = Descent(hyper_parameters.learning_rate)

    train_loss, train_acc = loss_and_accuracy(train_data, model, device)

    println("0 Epochen von $(hyper_parameters.training_steps): Loss ist $train_loss, Accuracy ist $train_acc.")
    clf()
    plot(train_losses, "b")
    
    for epoch in 1:hyper_parameters.training_steps
        for (x, y) in train_data
            x, y = device(x), device(y) # transfer data to device
            gs = gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        plot_loss(epoch, 20, train_data, model, device)
    end

    cpu(model)
    model |> cpu
    train_loss, train_acc = loss_and_accuracy(train_data, model, device)
    println("Loss: $train_loss, Accuracy: $train_acc")

    save_weights(model, "model.bson", train_losses)
    @info "Weights saved at \"model.bson\""
end

function anynan(c)
    for i1 = 1:size(c)[1], i2 = 1:size(c)[2]
        if isnan(c[i1, i2])
            println("NaN at [$i1, $i2]!")
        end
    end
end

function bla()
    confusion_matrix(get_loader(false), JLD2.load("mymodel.jld2")[:"model"])
end

mutable struct Args
    learning_rate :: Float64
    batch_size :: Int
    training_steps :: Float64
    shuffle :: Bool
end

global hyper_parameters = Args(0.008, 10, 500, true)

train(false)
end # module

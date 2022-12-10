using Flux, CUDA, ProgressMeter, TensorBoardLogger, Dates, BSON
using Flux.Zygote
mutable struct ModelParams
    config
    name::String
    # network_builder::Function
    η::Float32
    opt_type::DataType
    epochs::Int
    cuda::Bool
    loss_accuracy_portion::Float32
    prune_frq::Int
    prune_amount::Int
end

"""
Create ModelParams instance using given configuration file.
"""
function create_params(config)::ModelParams
    params = ModelParams(
        config,
        config.MODEL_NAME,
        # config.MODEL,
        config.LEARNING_RATE,
        config.OPTIMIZER,
        config.EPOCHS,
        config.USE_CUDA,
        config.LOSS_ACCURACY_PORTION,
        config.PRUNE_FREQUENCY,
        config.PRUNE_AMOUNT
    )
    return params
end

"""
Model instance.
"""
mutable struct EEGModel
    model::Flux.Chain
    params::ModelParams
    loss::Function
    noise::Function
    #   l2::Function
    opt
    device::Function
    epochs_done::Int
    epochs_goal::Int
    logger
    logger_name::String
    train_loss_history::Array{Float32,1}
    test_loss_history::Array{Float32,1}
    train_acc_history::Array{Float32,1}
    test_acc_history::Array{Float32,1}
    prune_guard::Array{Int8, 1}
end

sqnorm(x) = sum(abs2, x)
"""
Calculate loss.

This function is for training, for logging purposes loss_accuracy is recommended as it barely takes more performance.
"""
function loss(model::EEGModel, x, y)
    return (model.loss(model.model(x), y)) # + model.l2(model))
end

"""
Return CUDA device function (cpu or gpu).
"""
function get_device(params::ModelParams)::Function
    if params.cuda == true
        if CUDA.functional()
            return gpu
        else
            @warn "Despite USE_CUDA = true, CUDA is disabled as it isn't supported."
            return cpu
        end
    else
        return cpu
    end
end

function get_time_str()::String
    return Dates.format(now(), "YYYY-mm-dd_HH-MM-SS")
end

function no_noise(model, x)
    return x
end

function new_logger(params::ModelParams)
    time_string = get_time_str()
    logger_name = replace(params.name, "*" => time_string)
    logger = TBLogger("model-logging/$(logger_name)", tb_overwrite, prefix=time_string)
    return logger_name, logger
end

function valid_layer(layer)
    if (typeof(layer) <: Conv) || (typeof(layer) <: Dense)
        return true
    else
        return false
    end
end

function get_output_index(net::Flux.Chain)
    out = length(net.layers)
    while (!valid_layer(net.layers[out]) && out >= 1)
        out -= 1
    end
    return out
end

"""
Create new, independent model using given configuration.
"""
function new_model(config)::EEGModel
    loss = config.LOSS
    params = create_params(config)
    
    opt = params.opt_type(config.LEARNING_RATE)
    net = config.MODEL #params.network_builder()
    logger_name, logger = new_logger(params)
    dev = get_device(params)

    out_index = get_output_index(net)
    prune_guard = [out_index:length(net.layers)..., config.PRUNE_GUARD...]
    if config.NOISE
        noise_function = config.NOISE_FUNCTION
    else
        noise_function = no_noise
    end
    # if config.L2
    #     l2 = (model) -> return 0.0001 * sum(sqnorm, Flux.params(model.model))
    # else
    #     l2 = (model) -> return 0.0
    # end

    return EEGModel(net, params, loss, noise_function, opt,
        dev, 0, params.epochs, logger, logger_name, [], [], [], [], prune_guard)
end

"""
Save given model at given path. If path is empty, generate automatically.
"""
function save_model(model::EEGModel, path::String="")
    model.model = cpu(model.model)
    path = isempty(path) ? "model-logging/$(model.logger_name)/model.bson" : path
    bson(path, model=model)
    model.model = model.device(model.model)
end

function get_most_recent(dir_path)
    files = []
    for file in readdir(dir_path)
        if file[end-4:end] == ".bson"
            push!(files, file)
        end
    end
    times = []
    for file_path in files
        time_str = file_path[7:end-5]
        push!(times, DateTime(time_str, "YYYY-mm-dd_HH-MM-SS"))
    end
    file = files[argmax(times)]
    return dir_path * "/" * file
end

function load_model(path::String)::EEGModel
    if isdir(path)
        path = get_most_recent(path)
    end
    if isempty(path)
        throw(ArgumentError("Given path is empty!"))
    end
    data = BSON.load(path, @__MODULE__)
    model = data[:model]
    if model.epochs_done >= model.epochs_goal
        model.epochs_goal = model.epochs_done + model.params.epochs
    end
    return model
end
function load_model(config)
    return load_model(config.LOAD_PATH)
end

function fill_param_dict!(dict, m, prefix::String)
    fields2ignore = [:dilation, :groups, :pad, :stride, # Conv layers
        :k, # MaxPool
        :p, # Dropout
    ]
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix * "layer_" * string(i) * "/" * string(layer) * "/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            if !(fieldname in fields2ignore)
                val = getfield(m, fieldname)
                if val isa AbstractArray
                    val = vec(val)
                end
                dict[prefix*string(fieldname)] = val
            end
        end
    end
end

"""
Function to get dictionary of model parameters.
Copied with slight changes from https://docs.juliahub.com/TensorBoardLogger/bP1Zo/0.1.13/examples/flux/.
"""
function get_param_dict(model::EEGModel)
    m = model.model
    dict = Dict{String,Any}()
    fill_param_dict!(dict, m, "")
    return dict
end

"""
Calculate loss and accuracy at the same time, thereby minimizing performance cost.
"""
function loss_accuracy(model::EEGModel, data_loader::Flux.Data.DataLoader, name::String="")
    total = 0
    accurate = 0
    l = Float32(0)
    i = 0

    @showprogress 3 "    Calculating $(name) performance..." for (x, y) in (data_loader)
        i += 1
        if mod(i, (1.0 / model.params.loss_accuracy_portion)) == 0
            x, y = model.device(x), model.device(y)
            est = model.model(x)
            l += model.loss(est, y)
            same_result_bitarray = Flux.onecold(est) .== Flux.onecold(y)
            accurate += sum(same_result_bitarray)
            total += length(same_result_bitarray)
        end
    end
    return (loss=l / total, accuracy=accurate / total)
end

"""
Callback function to be executed for logging purposes
"""
function logging_cb(model::EEGModel, data::Data)
    param_dict = get_param_dict(model)
    testmode!(model)
    train_loss, train_acc = loss_accuracy(model, data.train_data, "train")
    test_loss, test_acc = loss_accuracy(model, data.test_data, "test")
    push!(model.train_loss_history, train_loss)
    push!(model.test_loss_history, test_loss)
    push!(model.train_acc_history, train_acc)
    push!(model.test_acc_history, test_acc)

    Base.with_logger(model.logger) do
        @info "model" params = param_dict log_step_increment = 0
        @info "train" loss = train_loss acc = train_acc log_step_increment = 0
        @info "test" loss = test_loss acc = test_acc
    end

    if test_acc > 0.6
        return true
    else
        return false
    end
end

function saving_cb(model::EEGModel)
    save_model(model, "model-logging/$(model.logger_name)/model_$(get_time_str()).bson")
end

function prune_cb!(model::EEGModel, data::Data)
    if (model.epochs_done % model.params.prune_frq) == 0
        if model.train_acc_history[end] > (model.test_acc_history[end] + 0.3)
            @info "Pruning!"
            model.model = model.device(prune(model, data))
        end
    end
    # CUDA.unsafe_free!(model.model)
end

"Only execute callback once every second."
throttled_log_cb = Flux.throttle(logging_cb, 10)
"Only execute callback once every second."
throttled_save_cb = Flux.throttle(saving_cb, 20)

"""
TODO
"""
function train_epoch!(model::EEGModel, data::Data, log=true, save=true)
    local train_loss::Float32
    local batch_losses::Array{Float32} = []
    local early_stop = Bool

    ps = Flux.params(model.model)
    model.model = model.device(model.model)
    # ps = Params(ps)
    # "Epoch $(model.epochs_done + 1): " 5 
    @showprogress "Epoch $(model.epochs_done + 1): " for (x, y) in (data.train_data)

        trainmode!(model.model)
        x = model.noise(model, model.device(x))
        # x = model.noise(model, x)
        y = model.device(y)
        # back is a method that computes the product of the gradient so far with its argument.
        train_loss, back = Zygote.pullback(() -> loss(model, x, y), ps)
        push!(batch_losses, train_loss)
        # Insert whatever code you want here that needs training_loss, e.g. logging.
        # logging_callback(training_loss)
        # Apply back() to the correct type of 1.0 to get the gradient of loss.

        gs = back(one(train_loss))
        Flux.Optimise.update!(model.opt, ps, gs)

        if save
            throttled_save_cb(model)
        end
        if log
            early_stop = throttled_log_cb(model, data)
            if early_stop
                println("Stop early!")
                break
            end
        end
    end
    avg_train_loss = sum(batch_losses) / length(batch_losses)
    push!(model.train_loss_history, avg_train_loss)
    model.epochs_done += 1
    # prune_cb!(model, data) # Because error
end

"""
Train network... TODO
"""
function train!(model::EEGModel, data::Data)
    # train_epoch!(model, data, false)
    # train_epoch!(model, data, false)
    try
        while model.epochs_done < model.epochs_goal
            train_epoch!(model, data)
        end
    catch e
        if typeof(e) == InterruptException
            @info "Noooo I was interrupted!"
            saving_cb(model)
            # Additional handling... ?
        else
            throw(e)
        end
    end
end
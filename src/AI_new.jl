# module AI
using Flux, DataFrames, ProgressMeter, TensorBoardLogger, CUDA, BSON
export ModelData, create_model, train!, save, load_model

struct ModelData
    dataloader::Flux.DataLoader
    tag2output::Dict{Symbol,Array{Number}}
    datadescriptor::DataDescriptor
end

function Base.getproperty(obj::ModelData, name::Symbol)
    if name in [:inputs, :outputs]
        return getfield(obj.dataloader.data, name)
    end
    return getfield(obj, name)
end

rowtovec(df::DataFrame, row::Int) = [df[row, i] for i in 1:ncol(df)]
rowtovec(df::SubDataFrame, row::Int) = [df[row, i] for i in 1:ncol(df)]

function rowstomatrix(df::DataFrame, row::Int, numrows::Int)
    result = Array{typeof(df[1, 1]),2}(undef, ncol(df), numrows)
    for i in 1:numrows
        result[:, i] = rowtovec(df, row + i - 1)
    end
    return result
end

function rowstomatrix(df::SubDataFrame, row::Int, numrows::Int)
    result = Array{typeof(df[1, 1]),2}(undef, ncol(df), numrows)
    for i in 1:numrows
        result[:, i] = rowtovec(df, row + i - 1)
    end
    return result
end

element(dict::Dict) = collect(dict)[1][2]

numrows(::FFTDataDescriptor) = 1
function inputarray(data_descriptor::FFTDataDescriptor, num_samples::Int)
    return Array{Float32,3}(undef,
        data_descriptor.num_channels,
        data_descriptor.max_freq,
        num_samples
    )
end
function rows2input(data_descriptor::FFTDataDescriptor, rows::Matrix)
    return reshape(rows,
        data_descriptor.num_channels,
        data_descriptor.max_freq,
        1
    )
end

numrows(::RawDataDescriptor) = 200
function inputarray(data_descriptor::RawDataDescriptor, num_samples::Int)
    return Array{Float32,3}(undef,
        data_descriptor.num_channels,
        numrows(data_descriptor),
        num_samples
    )
end
function rows2input(data_descriptor::RawDataDescriptor, rows::Matrix)
    return reshape(rows,
        data_descriptor.sample_width,
        numrows(data_descriptor),
        1
    )
end

"""
    ModelData(data::Data, outputs::Dict; batchsize=5, shuffle=true)

Create `ModelData` suitable for use with neural networks.
 
`data`: EEG data.

`tag2output`: Dictionary; each key is a tag (saved in data) with the value
representing the output the model should give for that tag.

Example:

```julia
# classification:
data_class = load_data("data/Emotions", "RawData")
tag2output_class = Dict(
    :scared => [1.0, 0.0, 0.0],
    :angry => [0.0, 1.0, 0.0],
    :happy => [0.0, 0.0, 1.0]
)
modeldata_class = ModelData(data, tag2output_class)
# regression:
data_reg = load_data("data/Concentration", "RawData")
tag2output_reg = Dict(
    :rest => [0.0],
    :task1 => [0.2],
    :task2 => [0.7],
    :task3 => [1.0]
)
modeldata_reg = ModelData(data, tag2output_reg)
```
"""
function ModelData(data::Data, tag2output::Dict; batchsize=5, shuffle=true)
    seperated_inputs = Array{SubDataFrame,1}(undef, length(tag2output))
    seperated_outputs = Array{valtype(tag2output),1}(undef, length(tag2output))

    for (i, tag) in enumerate(keys(tag2output))
        seperated_inputs[i] = @view data.df[in.(string(tag), data.df.tags), 4:end]
        seperated_outputs[i] = tag2output[tag]
    end

    num_samples = sum(size.(seperated_inputs, 1))
    outputs = Array{valtype(element(tag2output)),ndims(element(tag2output)) + 1}(undef, size(element(tag2output))..., num_samples)
    fill!(outputs, 0)

    inputs = inputarray(data.data_descriptor, num_samples)
    i = 1

    # TODO: speed up by using comprehensions, map?
    for (df, output) in zip(seperated_inputs, seperated_outputs)
        num_rows = numrows(data.data_descriptor)
        for row in 1:num_rows:size(df, 1)
            if (size(df, 1) - row) < num_rows
                break
            end
            # create view of inputs for this sample by indexing only last
            # dimension with i
            inputs_sample = selectdim(inputs, ndims(inputs), i)
            inputs_sample[:] = rows2input(data.data_descriptor, rowstomatrix(df, row, num_rows))
            # create view of outputs for this sample by indexing only last
            # dimension with i
            output_sample = selectdim(outputs, ndims(outputs), i)
            output_sample[:] = output
            i += 1
        end
    end
    return ModelData(
        Flux.DataLoader((inputs=inputs, outputs=outputs), shuffle=shuffle, batchsize=batchsize),
        tag2output, data.data_descriptor
    )
end

mutable struct Model
    network::Flux.Chain
    epochs::Int
    savedir::String
    modelname::String
end

function Base.:(==)(a::Flux.Chain, b::Flux.Chain)
    # check number of layers
    if length(a.layers) !== length(b.layers)
        return false
    end
    for (layer_a, layer_b) in zip(a.layers, b.layers)
        # check types of layers
        if typeof(layer_a) !== typeof(layer_b)
            return false
        end
        # check weights, biases, etc.
        for field in fieldnames(typeof(layer_a))
            if any(getfield(layer_a, field) .!== getfield(layer_b, field))
                return false
            end
        end
    end
    return true
end

Base.:(==)(a::Model, b::Model) =
    a.epochs == b.epochs && a.savedir == b.savedir &&
    a.modelname == b.modelname && a.network == b.network

# Overloading Flux Methods:

trainmode!(model::Model) = Flux.trainmode!(model.network)
# make model(inputs) = model.network(inputs)
(model::Model)(inputs) = model.network(inputs)

# e.g. make (10, 3, 7, 8) into (10, 3, 7, 1, 8)
add_dimension(layer) = reshape(layer, size(layer)[1:end-1], 1, size(layer)[end])

"""
    standard_network(inputshape, outputshape, datadescriptor)::Flux.Chain

Choose and create a neural network based on the given `inputshape`,
`outputsshape` and `datadescriptor`.

These networks are what we're currently using and may not be ideal for you.

There are only networks predefined for `RawDataDescriptor` and
`FFTDataDescriptor` with a 1d output.
"""
function standard_network(inputshape, outputshape, datadescriptor)::Flux.Chain
    # add dimension because of samples
    inputshape = (inputshape..., 1)
    if datadescriptor isa FFTDataDescriptor
        # only 1 output dimension
        if length(outputshape) == 1
            return @autosize (inputshape...,) Chain(
                Conv((5,), inputshape[2] => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 128, relu, pad=SamePad()),
                Conv((5,), 128 => 256, relu, pad=SamePad()),
                Conv((5,), 256 => 512, relu, pad=SamePad()),
                Conv((3,), 512 => outputshape[1]),
                Flux.flatten,
                Dense(_, outputshape[1])
            )
        end
        # TODO: error...?
    end
    if datadescriptor isa RawDataDescriptor
        # only 1 output dimension
        if length(outputshape) == 1
            return @autosize (inputshape...,) Chain(
                Conv((5,), _ => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 128, relu, pad=SamePad()),
                Conv((5,), 128 => 256, relu, pad=SamePad()),
                Conv((5,), 256 => 512, relu, pad=SamePad()),
                Conv((3,), 512 => outputshape[1]),
                Flux.flatten,
                Dense(_, outputshape[1])
            )
        end
    end

    throw(ArgumentError("No standard network structure defined for datadescriptor of type $(typeof(datadescriptor))!"))
end

function get_inputshape(data_desc::FFTDataDescriptor)
    return (data_desc.num_channels, data_desc.max_freq)
end

function get_timestr()::String
    return Dates.format(now(), "YYYY-mm-dd_HH-MM-SS")
end

function parse_modelname(modelname::String)
    timestr = get_timestr()
    return replace(modelname, "*t" => timestr)
end

function create_model(inputshape, outputshape, datadescriptor;
    network_constructor::Function=standard_network, savedir="models/", modelname="Standard_*t")::Model
    network = network_constructor(inputshape, outputshape, datadescriptor)
    epochs = 0
    modelname = parse_modelname(modelname)
    return Model(network, epochs, savedir, modelname)
end

"""
    create_model(data::ModelData; network_constructor::Function=standard_network)
    
Create a model with a neural network for classifying / regressing `data`.

`network_constructor`: The given function is given three arguments: the
input shape and output shape of the data, each represented by a tuple, and the
data descriptor of type DataDescriptor.

The default `network_constructor` if not supplied is [`standard_network`](@ref).
"""
function create_model(data::ModelData; network_constructor::Function=standard_network,
    savedir="models/", modelname="Standard_*t")::Model
    # end-1 because last dimension is along samples
    inputshape = size(data.inputs)[1:end-1]
    outputshape = size(data.outputs)[1:end-1]
    return create_model(inputshape, outputshape, data.datadescriptor,
        network_constructor=network_constructor, savedir=savedir, modelname=modelname)
end

function save(model::Model, path::String, overwrite=false)
    if !overwrite && isfile(path)
        throw(ErrorException("File $path already exists! Specify a different file or set overwrite to true!"))
    end
    if !isdir(dirname(path))
        createpath(dirname(path) * "/")
    end
    model.network = model.network |> cpu
    bson(path, model=model)
end

function save(model::Model)
    save(model, joinpath(model.savedir, model.modelname, "model_" * get_timestr() * ".bson"))
end

function get_most_recent(dir_path::String)::String
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

function load_model(path::String)::Model
    if isdir(path)
        path = get_most_recent(path)
    end
    if !isfile(path)
        throw("File $path doesn't exist!")
    end
    model = BSON.load(path, @__MODULE__)[:model]
    return model
end

mutable struct TrainingParameters
    # Learning rate.
    η::Float32
    # how many epochs to train.
    epochs::Int
    # a noise function to apply to inputs
    noise::Function
    # a device to move data to (`gpu` or `cpu`)
    device::Function
    # a loss function `f(ŷ, y)`` returning a single number
    loss::Function
    # optimiser to use; `Adam`, `GradientDescent` etc.
    opt_rule::Any
    # parameters to pass to optimiser, e.g. [β = 0.4]
    opt_params::Array
    # whether to show progress bar
    show_progress::Bool
    # interval for auto saving model; if negative, model won't be saved
    autosave::Integer
end


function standard_trainingparameters()
    η = 0.01
    epochs = 10
    noise = x -> x
    device = CUDA.functional() ? gpu : cpu
    loss = Flux.logitcrossentropy
    opt_rule = Adam
    # params to pass to opt_rule apart from η
    opt_params = []
    show_progress = true
    autosave = 5
    TrainingParameters(η, epochs, noise, device, loss, opt_rule, opt_params, show_progress, autosave)
end

function train!(model::Model, data::ModelData, params::TrainingParameters)
    # callbacks:
    if params.autosave >= 0
        _save_model(model::Model) = begin
            save(model)
            model.network = model.network |> params.device
        end
        saving_cb = Flux.throttle(_save_model, params.autosave)
    else
        saving_cb = () -> ()
    end
    epochgoal = model.epochs + params.epochs
    trainmode!(model)
    opt = Flux.setup(params.opt_rule(params.η, params.opt_params...), model.network)
    model.network = model.network |> params.device
    # TODO: performance considerations of try/catch
    try
        while model.epochs < epochgoal
            # move train_loss up into this scope
            local train_loss::eltype(data.dataloader.data.inputs)

            # init progress bar
            num_samples = size(data.dataloader.data.outputs, ndims(data.dataloader.data.outputs))
            p = Progress(num_samples, enabled=params.show_progress)
            i = 0

            for (x, y) in data.dataloader
                # move inputs and outputs to device and apply noise on inputs
                x = x |> params.device |> params.noise
                y = y |> params.device
                # calculate gradients
                train_loss, gs = Flux.withgradient(model.network, x) do network, xs
                    ŷ = network(xs)
                    return params.loss(ŷ, y)
                end
                # update weights
                Flux.update!(opt, model.network, gs[1])
                # update progress bar
                i += size(y, ndims(y))
                update!(p, i)
            end
            saving_cb(model)
            model.epochs += 1
        end
    catch e
        if typeof(e) == InterruptException
            @info "Handling Interruption..."
            # TODO: save model
        else
            throw(e)
        end
    end
end

function train!(model::Model, data::ModelData;
    params::TrainingParameters=standard_trainingparameters(), kws...)
    for key in keys(kws)
        if !hasfield(TrainingParameters, key)
            throw(ArgumentError("Given kwarg $key not recognized!"))
        end
        setproperty!(params, key, kws[key])
    end
    train!(model, data, params)
end

# end # module
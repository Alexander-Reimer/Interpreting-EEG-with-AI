# module AI
using Flux, DataFrames, ProgressMeter, TensorBoardLogger, CUDA
export ModelData, create_model, train!

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

element(dict::Dict) = collect(dict)[1][2]

function inputarray(data_descriptor::FFTDataDescriptor, num_samples::Int)
    return Array{Float32,3}(undef,
        data_descriptor.num_channels,
        data_descriptor.max_freq,
        num_samples
    )
end

function row2inputs(data_descriptor::FFTDataDescriptor, row::Vector)
    return reshape(row,
        data_descriptor.num_channels,
        data_descriptor.max_freq,
        1
    )
end

function inputarray(data_descriptor::RawDataDescriptor, num_samples::Int)
    return Array{Float32,2}(undef,
        data_descriptor.num_channels,
        num_samples
    )
end

function row2inputs(data_descriptor::RawDataDescriptor, row::Vector)
    return reshape(row,
        data_descriptor.sample_width,
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
        for row in 1:size(df, 1)
            # create view of inputs for this sample by indexing only last
            # dimension with i
            inputs_sample = selectdim(inputs, ndims(inputs), i)
            inputs_sample[:] = row2inputs(data.data_descriptor, rowtovec(df, row))
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

struct Model
    network::Flux.Chain
    epochs::Int
end

# Overloading Flux Methods:

trainmode!(model::Model) = trainmode!(model.network)
# make model(inputs) = model.network(inputs)
(model::Model)(inputs) = model.network(inputs)

# e.g. make (10, 3, 7, 8) into (10, 3, 7, 1, 8)
add_dimension(layer) = reshape(layer, size(layer)[1:end-1], 1, size(layer)[end])

function standard_network(inputshape, outputshape, datadescriptor)::Flux.Chain
    if datadescriptor isa FFTDataDescriptor
        # only 1 output dimension
        if length(outputshape) == 1
            return Chain(
                Conv((5,), inputshape[2] => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 128, relu, pad=SamePad()),
                Conv((5,), 128 => 256, relu, pad=SamePad()),
                Conv((5,), 256 => 512, relu, pad=SamePad()),
                Conv((3,), 512 => outputshape[1]),
                Flux.flatten,
            )
        end
    end
    if datadescriptor isa RawDataDescriptor
        # only 1 output dimension
        if length(outputshape) == 1
            return Chain(
                Conv((5,), 60 => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 64, relu, pad=SamePad()),
                Conv((5,), 64 => 128, relu, pad=SamePad()),
                Conv((5,), 128 => 256, relu, pad=SamePad()),
                Conv((5,), 256 => 512, relu, pad=SamePad()),
                Conv((3,), 512 => outputshape[1]),
                Flux.flatten,
            )
        end
    end

    throw(ArgumentError("No standard network structure defined for datadescriptor of type $(typeof(datadescriptor))!"))
end

function get_inputshape(data_desc::FFTDataDescriptor)
    return (data_desc.num_channels, data_desc.max_freq)
end

function create_model(inputshape, outputshape, datadescriptor;
    network_constructor::Function=standard_network)::Model
    network = network_constructor(inputshape, outputshape, datadescriptor)
    epochs = 0
    return Model(network, epochs)
end

"""
    create_model(data::ModelData; network_constructor::Function=standard_network)
    
Create a model with a neural network for classifying / regressing `data`.

`network_constructor`: The given function is given three arguments: the
input shape and output shape of the data, each represented by a tuple, and the
data descriptor of type DataDescriptor.
"""
function create_model(data::ModelData; network_constructor::Function=standard_network)::Model
    # end-1 because last dimension is along samples
    inputshape = size(data.inputs)[1:end-1]
    outputshape = size(data.outputs)[1:end-1]
    return create_model(inputshape, outputshape, data.datadescriptor,
        network_constructor=network_constructor)
end

mutable struct TrainingParameters
    # Learning rate.
    η::Float32
    epochs::Int
    noise::Function
    device::Function
    loss::Array{Function}
end

function standard_trainingparameters()
    η = 0.01
    epochs = 10
    noise = x -> x
    device = CUDA.functional() ? gpu : cpu
    loss = [Flux.logitcrossentropy]
    TrainingParameters(η, epochs, noise, device, loss)
end

function train_epoch!(model::Model, data::ModelData, params::TrainingParameters, epochgoal::Int)
    local training_loss
    ps = Flux.Params()
    for (x, y) in data.dataloader
        x = x |> params.device |> params.noise
        y = y |> params.device
        gs = gradient(ps) do
            training_loss = loss()
        end
    end
end

Flux.train!
function train!(model::Model, data::ModelData, params::TrainingParameters)
    try
        epochgoal = model.epochs + params.epochs
        trainmode!(model)
        while model.epochs < epochgoal

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
    parameters::Function=standard_trainingparameters, kws...)
    params::TrainingParameters = parameters()
    for key in keys(kws)
        if !hasfield(TrainingParameters, key)
            throw(ArgumentError("Given kwarg $key not recognized!"))
        end
        setproperty!(params, key, kws[key])
    end
end

# end # module
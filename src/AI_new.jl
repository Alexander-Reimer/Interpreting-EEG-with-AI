# module AI
using Flux, DataFrames

struct ModelData
    dataloader::Flux.DataLoader
    tag2output::Dict{Symbol,Array{Number}}
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

"""
    ModelData(data::Data, outputs::Dict; batchsize=5, shuffle=true)

Create `ModelData` suitable for use with neural networks.
 
`data`: 

`tag2output`:
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

    if data.data_descriptor isa FFTDataDescriptor
        inputs = Array{Number,3}(undef,
            data.data_descriptor.num_channels,
            data.data_descriptor.max_freq,
            num_samples
        )
        i = 1
        for (df, output) in zip(seperated_inputs, seperated_outputs)
            for row in 1:size(df, 1)
                inputs[:, :, i] = reshape(rowtovec(df, row),
                    data.data_descriptor.num_channels,
                    data.data_descriptor.max_freq,
                    1
                )
                outputs[:, :, i] = output
                i += 1
            end
        end
        return ModelData(
            Flux.DataLoader((inputs=inputs, label=outputs), shuffle=shuffle, batchsize=batchsize),
            outputs)
    end

    if data.data_descriptor isa RawDataDescriptor
        inputs = Array{Number,2}(undef, data.data_descriptor.sample_width, num_samples)
        i = 1
        for (out, df) in zip(seperated_outputs, seperated_inputs)
            for row in 1:size(df, 1)
                inputs[:, i] = reshape(rowtovec(df, row),
                    data.data_descriptor.sample_width,
                    1
                )
                @info size(outputs[:, i])
                outputs[:, i] = out
                i += 1
            end
        end
        data_loader = Flux.DataLoader((inputs=inputs, outputs=outputs), shuffle=shuffle, batchsize=batchsize)
        return ModelData(data_loader, tag2output)
    end
end

struct Model
    network::Flux.Chain
    # loss::Function
end

# e.g. make (10, 3, 7, 8) into (10, 3, 7, 1, 8)
add_dimension(layer) = reshape(layer, size(layer)[1:end-1], 1, size(layer)[end])

function default_fft(input_shape, output_shape)
    @info input_shape
    if (length(input_shape) < 3)
        # e.g. make (10, 5) into (10, 1, 5)
        input_shape = (input_shape[1:end-1], 1, input_shape[end])
    end
    return @autosize (input_shape...,) Chain(
        Conv((5,), 1 => 64, pad=SamePad(), relu),
        Conv((5,), 64 => 64, pad=SamePad(), relu),
        Conv((5,), 64 => 128, pad=SamePad(), relu),
        Conv((5,), 128 => 256, pad=SamePad(), relu),
        Conv((5,), 256 => 512, pad=SamePad(), relu),
        Conv((3,), 512 => output_shape),
        Flux.flatten,
    )
end

function standard_networks(input_shape, output_shape, datadescriptor)
    
end

function get_input_shape(data_desc::FFTDataDescriptor)
    return (data_desc.num_channels, data_desc.max_freq)
end

function create_model(input_shape, output_shape; network_constructor::Function=default_network)
    network = network_constructor(input_shape, output_shape)
    return Model(network)
end

function create_model(data::ModelData; network_constructor::Function=default_network)
    # end-1 because last dimension is along samples
    input_shape = size(data.inputs)[1:end-1]
    output_shape = size(data.outputs)[1:end-1]
    return create_model(input_shape, output_shape, network_constructor=network_constructor)
end

# end # module
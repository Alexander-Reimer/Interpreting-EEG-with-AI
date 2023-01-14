# module AI
using Flux, DataFrames

struct ModelData
    dataloader::Flux.DataLoader
    outputs::Dict{Symbol,Array{Number}}
end

rowtovec(df::DataFrame, row::Int) = [df[row, i] for i in 1:ncol(df)]
rowtovec(df::SubDataFrame, row::Int) = [df[row, i] for i in 1:ncol(df)]

somedictelement(dict::Dict) = collect(dict)[1][2]

"""
    ModelData(data::Data, outputs::Dict; batchsize=5, shuffle=true)

Create data loader.
outputs must be {Symbol,AbstractArray{Number}}
"""
function ModelData(data::Data, outputs::Dict; batchsize=5, shuffle=true)
    dfs = Array{SubDataFrame,1}(undef, length(outputs))
    out_type = typeof(somedictelement(outputs))
    dfs_out = Array{out_type,1}(undef, length(outputs))
    for (i, tag) in enumerate(keys(outputs))
        dfs[i] = @view data.df[in.(string(tag), data.df.tags), 4:end]
        dfs_out[i] = outputs[tag]
    end
    num_samples = sum(size.(dfs, 1))
    labels = Array{out_type,1}(undef, num_samples)
    if data.data_descriptor isa FFTDataDescriptor
        inputs = Array{Number,3}(undef,
            data.data_descriptor.num_channels,
            data.data_descriptor.max_freq,
            num_samples
        )
        i = 1
        for (out, df) in zip(dfs_out, dfs)
            for row in 1:size(df, 1)
                inputs[:, :, i] = reshape(rowtovec(df, row),
                    data.data_descriptor.num_channels,
                    data.data_descriptor.max_freq,
                    1
                )
                labels[i] = out
                i += 1
            end
        end
    end
    return ModelData(
        Flux.DataLoader((data=inputs, label=labels), shuffle=shuffle, batchsize=batchsize),
        outputs)
end

struct Model
    train_data::ModelData
    test_data::ModelData
    network::Flux.Chain
end

function default_network(input_shape, output_shape)
    return @autosize (3, 4) Chain(
        Conv((5, 1), _ => 64, pad=SamePad(), relu),
        Conv((5, 1), 64 => 64, pad=SamePad(), relu),
        Conv((5, 1), 64 => 128, pad=SamePad(), relu),
        Conv((5, 1), 128 => 256, pad=SamePad(), relu),
        Conv((5, 1), 256 => 512, pad=SamePad(), relu),
        Conv((3, 1), 512 => output_shape),
        Flux.flatten,
    )
end

function get_input_shape(data_desc::FFTDataDescriptor)
    return (data_desc.num_channels, data_desc.max_freq)
end

function create_model(data::ModelData; network_constructor::Function=default_network,
    cases=[:left, :none, :right])
    input_shape = get_input_shape(train_data.data_descriptor)
    output_shape = length(cases)
    network = default_network(input_shape, output_shape)
end

# end # module
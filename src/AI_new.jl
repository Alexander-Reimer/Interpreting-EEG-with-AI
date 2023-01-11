module AI
using Flux, DataFrames

struct ModelData
    data::Array{SubDataFrame,1}
    outputs::Dict{Symbol,Array{Number}}
end

function getRealSample(df::SubDataFrame, data_desc::EEG.FFTDataDescriptor)
    select
end

rowtovec(df::DataFrame, row::Int) = [df[row, i] for i in 1:ncol(df)]

function ModelData(data::Data, outputs::Dict{Symbol,Array{Number}}; batchsize=5, shuffle=true)
    dfs = Array{SubDataFrame,1}(undef, length(outputs))
    dfs_out = Array{Array{Number},1}(undef, length(outputs))
    for (i, tag) in enumerate(keys(outputs))
        dfs[i] = @view data.df[in.(data.df.tags, tag), 4:end]
        dfs_out[i] = outputs[tag]
    end
    num_samples = sum(size.(dfs, 1))
    if data.data_descriptor <: EEG.FFTDataDescriptor
        inputs = Array{Number,3}(undef,
            data.data_descriptor.num_channels,
            data.data_descriptor.max_freq,
            num_samples
        )
        labels = Array{Array{Number},1}(undef, num_samples)
        i = 1
        for (out, df) in zip(dfs_out, dfs)
            for row in df
                inputs[:, :, i] = reshape(row[4:end],
                    data.data_descriptor.max_freq,
                    data.data_descriptor.num_channels)
                labels[i] = out
            end
        end
        return Flux.DataLoader((data=inputs, label=labels), shuffle = shuffle, batchsize = batchsize)
    end
end

struct Model
    train_data::ModelData
    test_data::ModelData
    network::Flux.Chain
end

function default_network(input_shape, output_shape)
    return @autosize (input_shape...) Chain(
        Conv((5, 1), _ => 64, pad=SamePad(), relu),
        Conv((5, 1), 64 => 64, pad=SamePad(), relu),
        Conv((5, 1), 64 => 128, pad=SamePad(), relu),
        Conv((5, 1), 128 => 256, pad=SamePad(), relu),
        Conv((5, 1), 256 => 512, pad=SamePad(), relu),
        Conv((_, 1), 512 => output_shape),
        Flux.flatten,
    )
end

function get_input_shape(data_desc::FFTDataDescriptor)
    return (data_desc.num_channels, data_desc.max_freq)
end

function create_model(train_data; network_constructor::Function=default_network,
    cases=[:left, :none, :right])
    input_shape = get_input_shape(train_data.data_descriptor)
    output_shape = length(cases)
    network = default_network(input_shape, output_shape)
end

end # module
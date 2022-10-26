"""
Get dimensions of activations after passing through each layer.

Return array with the sizes / size tuples.
"""
function act_dims(model::EEGModel, data::Flux.Data.DataLoader)
    # TODO: generalize
    activations = data.data[1][:, :, :, 1:1]
    activation_sizes = []
    # Move sample and model to gpu if activated
    activations = model.device(activations)
    model.model = model.device(model.model)
    # Move activations through each layer,
    # logging dimensions after each layer
    for layer in model.model
        activations = layer(activations)
        push!(activation_sizes, size(activations))
    end

    return activation_sizes
end

"""
Return sum of given array avg_activations and average of all activations of given data.
"""
function set_activations(model::EEGModel, avg_activations::Array, data::Flux.Data.DataLoader)
    avg_activations = model.device(avg_activations)
    num_samples = size(data.data[1])[end]
    for (activations, _) in data
        activations = model.device(activations)
        for (layer_i, layer) in enumerate(model.model.layers)
            layer = model.device(layer)
            activations = layer(activations)
            batch_dim = ndims(activations) # determine last dimension (batch)
            activations = sum(activations, dims=batch_dim) # calculate sum over all batches
            avg_activations[layer_i] .+= activations # add this sum
        end
    end
    avg_activations ./= num_samples # calculate average
    return avg_activations
end

"""
Return the average activations output by each layer of the model.
"""
function get_avg_activations(model::EEGModel, data::Flux.Data.DataLoader)
    testmode!(model.model)
    dims = act_dims(model, data)

    # Init arrays
    activations = Array{Array{Float32}}(undef, length(dims))
    for (i, layer_size) in enumerate(dims)
        activations[i] = zeros(Float32, layer_size...)
    end

    activations = set_activations(model, activations, data)
    return activations
end

function remove(layer::Conv; input_shape::Union{Int, Nothing} = nothing, index::Union{Int, Nothing} = nothing, sample = nothing)
    layer = cpu(layer)
    dims = size(layer.weight)

    if (index === nothing) && (input_shape === nothing)
        return layer, dims[4]
    end
    
    new_weights = layer.weight
    if input_shape === nothing
        input_channels = dims[3]
    else
        @assert input_shape <= dims[3]
        input_channels = input_shape
        new_weights = new_weights[:, :, 1:input_channels, :]

    end
    if index === nothing
        output_channels = dims[4]
    else
        @assert index <= dims[4]
        output_channels = dims[4] - 1
        new_weights = cat(new_weights[:, :, :, 1:index-1], new_weights[:, :, :, index+1:end], dims=4)
    end
    
    # TODO: can result of SamePad change with smaller number of out channels?
    kernel = dims[1:2]
    pad = layer.pad
    new_layer = Conv(kernel, input_channels => output_channels, pad = pad, layer.σ)
    new_layer.weight .= new_weights
    
    return new_layer, output_channels
end

function remove(layer::Dense; input_shape::Union{Int, Nothing} = nothing, index::Union{Int, Nothing} = nothing, sample = nothing)
    layer = cpu(layer)
    dims = size(layer.weight)

    if (index === nothing) && (input_shape === nothing)
        return layer, dims[1]
    end
    
    new_weights = layer.weight
    if input_shape === nothing
        inputs = dims[2]
    else
        @assert input_shape <= dims[2]
        inputs = input_shape
        new_weights = new_weights[:, 1:input_shape]
        
    end
    if index === nothing
        outputs = dims[1]
    else
        @assert index <= dims[1]
        outputs = dims[1] - 1
        new_weights = vcat(new_weights[1:outputs-1, :], new_weights[outputs+1:end, :])
    end
    
    # TODO: can result of SamePad change with smaller number of out channels?
    new_layer = Dense(inputs, outputs, layer.σ)
    new_layer.weight .= new_weights
    
    return new_layer, outputs
end

function remove(layer::MaxPool; input_shape = nothing, index = nothing, sample = nothing)
    if input_shape === nothing
        @error "Input shape can't be nothing for MaxPool"
    end
    # k = layer.k
    # output_shape = floor(Int, input_shape / k[1])
    output_shape = input_shape
    return layer, output_shape
end

function remove(layer::Function; input_shape = nothing, index = nothing, sample = nothing)
    if layer == flatten
        if sample === nothing
            @error "Sample needs to be given for flatten layer"
        end
        output_shape = length(sample) # total number of elements because flatten
        return layer, output_shape
    else
        @error "Unknown layer: $(layer)!"
    end
end

function remove(layer; input_shape = nothing, indexes::AbstractArray, sample = nothing)
    local out
    for index in indexes
        layer, out = remove(layer, input_shape = input_shape, index = index, sample = sample)
    end
    return layer, out
end


function get_layer_overfit(layer::Conv, difference, f)
    s = sum(difference, dims=[1,2,4])
    s = reshape(s, :)
    # TODO: update comments
    # calculate average difference in activations to
    # account for different sized layers
    overfit = f(s)
    # calculate output dimension with highest overfit
    index = argmax(s)
    return overfit, index
end

function get_layer_overfit(layer::Dense, difference, f)
    # s = sum(difference, dims=[1,2,4]) # not necessary for Dense?
    s = reshape(difference, :)
    # calculate average difference in activations to
    # account for different sized layers
    overfit = f(s)
    # calculate output dimension with highest overfit
    index = argmax(s)
    return overfit, index
end

function get_layer_overfit(layer::MaxPool, difference, f)
    return 0.0, 0
end

function get_layer_overfit(layer::Function, difference, f)
    if layer in [flatten]
        return 0.0, 0
    else
        @error "Unknown layer: $(layer)!"
    end
end

function average(x)
    return sum(x) / length(x)
end
"""
Determine how much neurons are overfitting.
"""
function determine_overfit(model::EEGModel, data::Data, f)
    train_avg = get_avg_activations(model, data.train_data)
    test_avg = get_avg_activations(model, data.test_data)
    differences = [abs.(diff) for diff in (train_avg .- test_avg)]
    
    overfits = Float32[]
    indexes = Int[]

    for (layer_index, layer) in enumerate(model.model.layers)
        if layer_index in model.prune_guard
            overfit, overfit_index = 0.0, 0
        else
            overfit, overfit_index = get_layer_overfit(layer, differences[layer_index], f)
        end
        push!(overfits, overfit)
        push!(indexes, (overfit_index))
    end
    return overfits, indexes
end

function prune(model::EEGModel, data::Data)
    global overfits
    model.model = model.device(model.model)
    new_chain = []
    overfits, indexes = determine_overfit(model, data, average)
    @debug "Overfits determined"
    sample = data.test_data.data[1][:, :, :, 1:1]
    output_shape = nothing
    for (i, layer) in enumerate(model.model.layers)
        @debug "Output shape", output_shape
        @debug "Layer ", i, layer
        if (i in model.prune_guard) || (overfits[i] < 0.0)
            @debug "Skipped layer ", i
            new_layer, output_shape = remove(layer, input_shape = output_shape, sample = sample)
            push!(new_chain, new_layer)
        else
            new_layer, output_shape = remove(layer, input_shape = output_shape, index = indexes[i], sample = sample)
            push!(new_chain, new_layer)
        end
        sample = new_layer(sample)
    end
    return Chain(new_chain...)
end
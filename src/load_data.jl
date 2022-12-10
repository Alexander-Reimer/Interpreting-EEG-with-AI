using Flux, PyCall
np = pyimport("numpy")

mutable struct Data
    train_data::Flux.Data.DataLoader
    test_data::Flux.Data.DataLoader
end

function load_file(path)
    return np.load(path)
end

function ignore_file(path)
    # Return true if file should be ignored
    name = split(path, "/")[end]
    if name[1] == "-"
        return true
    else
        return false
    end
end

function get_data_length(data_config)
    samples = 0
    for classification in data_config
        for file in readdir(classification[1])
            path = classification[1] * file
            if ignore_file(path) == false
                d = mapslices(rotr90, load_file(path), dims=[1, 2])
                d = mapslices(rotr90, d, dims=[2, 3])
                # d = mapslices(rotr90, d, dims=[1, 3])
                samples += size(d)[3]
            end
        end
    end
    return samples
end

function clip_scale(data)
    data = clamp.(data, -10, 10) ./ 10
    return data
end

function set_all_data!(X_data, Y_data, data_config, channels)
    i = [1, 0]
    for classification in data_config
        for file in readdir(classification[1])
            path = classification[1] * file
            if ignore_file(path) == false
                d = mapslices(rotr90, load_file(path), dims=[1, 2])
                d = mapslices(rotr90, d, dims=[2, 3])
                # d = mapslices(rotr90, d, dims=[1, 3])
                d = clip_scale(d)
                i[2] += size(d)[3]
                for j = 1:length(channels)
                    X_data[j, 1, :, i[1]:i[2]] = d[channels[j], :, :, :]
                end
                # println("i1: $(i[1]), i2: $(i[2])")
                # println(size(Y_traindata[:, i[1]:i[2]]))
                Y_data[:, i[1]:i[2]] .= classification[2]
                i[1] += size(d)[3]
            end
        end
    end
end

function load_data(config, type::Symbol)
    if type == :train
        data_config = config.TRAIN_DATA
    elseif type == :test
        data_config = config.TEST_DATA
    else
        @error "data type $type unknown! Only :test and :train supported!"
    end
    num_samples = get_data_length(data_config)
    num_outputs = length(data_config[1][2])
    
    # Pre-initialise arrays to improve performance
    X_data = Array{Float32}(undef, length(config.CHANNELS_USED), 1, config.MAX_FREQUENCY, num_samples)
    Y_data = Array{Float32}(undef, num_outputs, num_samples)

    set_all_data!(X_data, Y_data, data_config, config.CHANNELS_USED)

    return X_data, Y_data
end

function get_data(config)
    # TODO: Add sanity checks (same output number etc.)
    X_traindata, Y_traindata = load_data(config, :train)
    X_testdata, Y_testdata = load_data(config, :test)
    # Turn data into Flux DataLoaders
    train_data = Flux.Data.DataLoader((X_traindata, Y_traindata), batchsize=config.BATCH_SIZE, shuffle=config.SHUFFLE, partial=false)
    test_data = Flux.Data.DataLoader((X_testdata, Y_testdata), batchsize=config.BATCH_SIZE, shuffle=config.SHUFFLE, partial=false)

    # Clear unnecessary data
    X_traindata = nothing
    Y_traindata = nothing
    X_testdata = nothing
    Y_testdata = nothing

    return Data(train_data, test_data)
end
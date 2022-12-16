module EEG
export Device
export Experiment, gather_data!, save_data
include("EEG_Devices.jl")

"""
Abstract class containing data processors.

Defined data processors:
`StandardProcessor`: The standard processors with preset arguments and functions, for details see [`StandardProcessor`](@ref).
"""
abstract type DataProcessor end
using BrainFlow, DataFrames, Dates, CSV
import PyPlot
Plt = PyPlot
Plt.pygui(true)

# Utilitiy functions:
datetime(seconds_since_epoch) = Dates.unix2datetime(seconds_since_epoch)

"""
Single Channel with voltages in Volt and the times the voltages were recorded at in seconds.
"""
mutable struct Channel
    times::Array{Float64}
    voltages::Array{Float64}
end

"""
Device for gathering EEG data. Create it using
    
Device(board::EEGBoard)

... TODO
"""
mutable struct Device
    board::EEGBoard
    channels::Array{Channel}
    session_start::Float64
end

function Device(board::EEGBoard)
    channels = Array{Channel,1}(undef, board.num_channels)
    for i = 1:board.num_channels
        channels[i] = Channel([], [])
    end
    return Device(board, channels, time())
end

function update_data!(device::Device)
    for channel_num in eachindex(device.channels)
        channel = device.channels[channel_num]
        push!(channel.voltages, get_voltage(device.board, channel_num))
        push!(channel.times, (time() - device.session_start) * 1000) # convert from s to ms
    end
    return
end

"""
Standard configuration for processing EEG data. It uses a preset of functions and options and may not work for you.
Created using [`Standard`](@ref).

    Standard()
"""
struct StandardProcessor <: DataProcessor
    output_type::Dict{Symbol,Any}
    fft_maxfreq::Integer
end
"""
    Standard()::StandardProcessor

Create standard configuration for processing EEG data. See [`StandardProcessor`](@ref) for more details.
"""
function Standard()
    output_type = Dict()
    output_type[:type] = :SpectralDensity # requires output dims (num_channels, max_freq)
    processor = StandardProcessor(output_type, 60)
    return processor
end

"""
TODO: Processing

    1. Artifacts Removal
    2. Data Filtering
    3. Feature Extraction (FFT)
Also see [this](https://www.sciencedirect.com/science/article/pii/S1877705812022114/pdf?md5=3c7a2bdf5717d518cf46c4ef5d145d33&pid=1-s2.0-S1877705812022114-main.pdf), page 2529.
"""
function process(device::Device, processor::StandardProcessor)
    result = Array{Float64,2}(undef, device.board.num_channels, processor.fft_maxfreq)
    # Temporarily skip all processing
    # TODO
    for i in eachindex(result)
        result[i] = rand()
    end
    return result
end

mutable struct Data
    df::DataFrame
    metadata::DataFrame
    name::String
end

struct Metadata
    df::DataFrame
end

# For `meta.name` syntax (e.g. meta.num_channels)
function getproperty(meta::Metadata, name::Symbol)
    if name == :df
        return getfield(meta, name)
    else
        meta.df[1, name]
    end
end

# For `meta.name = value` syntax
function setproperty!(meta::Metadata, name::Symbol, value)
    if name == :df
        setfield!(meta, name, value)
    else
        meta.df[1, name] = value
    end
end

"""
Create new `Data`-Object for raw data.
"""
function Data(name, num_channels)
    column_names = ["time", "tags", "extraInfo"]
    for channel = 1:num_channels
        push!(column_names, string(channel))
    end
    types = fill(Float64[], num_channels)
    #        Time       tags                      extra_info    voltages  
    types = [Float64[], Array{Array{String},1}(), Dict[], types...]
    df = DataFrame(types, column_names)
    # column_names =  ["version",     "num_samples",  "column_names"]
    # types =         [VersionNumber[], Integer[],     AbstractArray{String}[]]
    metadata = DataFrame((
        version=v"0.0.0",
        num_samples=0,
        column_names=column_names
    ))
    return Data(df, metadata, name)
end

struct Experiment
    device::Device
    raw_data::Data
    tags::Array{String,1}
    extra_info::Dict{Symbol,Any}
    folderpath::String
    # cases::Array{Symbol,1}
end


"""
`name`: Name of the experiment (e.g. "BlinkDetection").

`tags`: Tags which will be applied to all data gathered with this Experiment.

`path`: To top-level of data directory (e.g. "data/"). If empty, files can't be saved.

TODO: `load_previous` not implemented yet (maybe in another function?)
"""
function Experiment(device::Device, name::String; tags::Array=[], extra_info::Dict=Dict(), path::String="data/", load_previous::Bool=false)
    folderpath = joinpath(path, name, "")
    raw_data = Data("RawData", device.board.num_channels)
    experiment = Experiment(device, raw_data, string.(tags), extra_info, folderpath)
    return experiment
end

function get_voltages(board::EEGBoard)
    voltages = Array{Float64,1}(undef, board.num_channels)
    for channel in eachindex(voltages)
        voltages[channel] = get_voltage(board, channel)
    end
    return time(), voltages
end

"""
    gather_data!(experiment::Experiment, runtime::Number; tags::Array=[], extra_info::Dict=Dict())

Gather raw EEG data. `runtime` is in seconds.

TODO: create test
"""
function gather_data!(experiment::Experiment, runtime::Number; tags::Array=[], extra_info::Dict=Dict(), delay::Number=0, save::Bool=true)
    all_tags = vcat(experiment.tags, string.(tags))
    new_row = Array{Any,1}(undef, 3 + experiment.device.board.num_channels)
    new_row[2] = all_tags
    new_row[3] = merge(experiment.extra_info, extra_info)
    start_time = time()
    new_num = 0
    if save == true
        save_data(empty(experiment.data), experiment.folderpath, updatemeta = false)
    end
    save_data(empty(experiment.data), experiment.folderpath, checkmeta = false, updatemeta = false)
    while (time() - start_time) < runtime
        new_row[1] = time()
        for channel = 1:experiment.device.board.num_channels
            new_row[3+channel] = get_voltage(experiment.device.board, channel)
        end
        push!(experiment.raw_data, new_row)
        if save
            sa
            CSV.write(experiment.path, experiment.raw_data[end:end, :], append=true)
        end
        if delay != 0
            sleep(delay)
        end
        new_num += 1
    end
    experiment.raw_data.metadata.num_samples += new_num
end

"""
Check if `df` is "compatible" with `metadata`.
"""
function is_compat(df::DataFrame, metadata::Metadata)
    if metadata.column_names == names(df)
        return true
    end
    return false
end

function combine_metadata(meta1::Metadata, meta2::Metadata)
    
end

function save_data(data::Data, folderpath; checkmeta = true, updatemeta = true)
    datapath = joinpath(folderpath, data.name * ".csv")
    metapath = joinpath(folderpath, data.name * "Metadata.csv")
    if checkmeta && isfile(metapath)
        metadata = CSV.read(metapath)
        if !is_compat(data.df, metadata)
            @error "Data not compatible with DataFrame at $file_path 
            (perhaps processed vs non-processed data or different 
            number of channels). If you want to overwrite the old data,
            delete both $file_path and $meta_file_path."
        end
    end
    if updatemeta
        CSV.write(metapath, data.metadata.df)
    end
    if !isempty(data.df)
        CSV.write(datapath, data.df, append=true)
    end
end

function save_data(experiment::Experiment)
    save_data(experiment.data, experiment.folderpath)
end

"""
Create `Data`-Object with a name, path, and a dataframe.
"""
function Data(df::DataFrame, metadata::Metadata, name, folder_path)
    file_path = joinpath(folder_path, name * ".csv")
    meta_file_path = joinpath(folder_path, name * "Metadata" * ".csv")
    if isfile(meta_file_path)
        metadata = CSV.read(meta_file_path, DataFrame)
        if !is_compat(df, metadata)
            @error "Given df differs structurally from already present 
            data at $file_path (perhaps processed vs non-processed data
            or different number of channels). If you want to overwrite,
            delete both $file_path and $meta_file_path."
        end
    end
    if !isfile(file_path)
        mkpath(dirname(file_path)) # create directories containing file
        io = open(basename(file_path), create=true) # create file
        close(io)
        CSV.write(file_path, df)
    end
    return Data(df, name, folder_path)
end

"""
Create `Data`-Object with a name, path, and a dataframe.
"""
function Data(df::DataFrame, metadata::Metadata, name, folder_path)
    file_path = joinpath(folder_path, name * ".csv")
    meta_file_path = joinpath(folder_path, name * "Metadata" * ".csv")
    if isfile(meta_file_path)
        metadata = CSV.read(meta_file_path, DataFrame)
        if !is_compat(df, metadata)
            @error "Given df differs structurally from already present 
            data at $file_path (perhaps processed vs non-processed data
            or different number of channels). If you want to overwrite,
            delete both $file_path and $meta_file_path."
        end
    end
    if !isfile(file_path)
        mkpath(dirname(file_path)) # create directories containing file
        io = open(basename(file_path), create=true) # create file
        close(io)
        CSV.write(file_path, df)
    end
    return Data(df, name, folder_path)
end
function get_sample()

end

mutable struct DataFilter
    include_tags::Array{String,1}
    exclude_tags::Array{String,1}
    extra_info_filter::Function
end

struct DataHandler
    data_processor::DataProcessor
    data_io#::DataIO
    cases::Union{Array{Symbol,1},Nothing}
    name::Union{String,Nothing}
    max_freq::Union{Int,Nothing}
end

"""
    DataHandler(data_processor::DataProcessor, data_io::DataIO; cases=nothing, name=nothing, max_freq=nothing)

Create a DataHandler instance. `cases`, `name` and `max_freq` are automatically determined by
data saved at path if they are `nothing`.

Example:

    data_io = DataIO("data/test", states)
    data_handler = DataHandler(data_io, StandardFFT())
"""
function DataHandler(data_processor::DataProcessor, data_io; cases=nothing, name=nothing, max_freq=nothing)

end
end
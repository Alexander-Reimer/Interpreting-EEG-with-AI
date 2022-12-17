module EEG
export Device
export Experiment, gather_data!, save_data
include("EEG_Devices.jl")

"""
Abstract class containing data processors.

Defined data processors:
`StandardProcessor`: The standard processors with preset arguments and functions, for details 
see [`StandardProcessor`](@ref).
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
Standard configuration for processing EEG data. It uses a preset of functions and options 
and may not work for you.

Create with [`Standard`](@ref).
"""
struct StandardProcessor <: DataProcessor
    output_type::Dict{Symbol,Any}
    fft_maxfreq::Integer
end
"""
    Standard()::StandardProcessor

Create standard configuration for processing EEG data. See [`StandardProcessor`](@ref)
for more details.
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
Also see [this](https://www.sciencedirect.com/science/article/pii\
/S1877705812022114/pdf?md5=3c7a2bdf5717d518cf46c4ef5d145d33&pid=1-s2\
.0-S1877705812022114-main.pdf), page 2529.
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

struct Metadata
    df::DataFrame
end

# For `meta.name` syntax (e.g. meta.num_channels)
function Base.getproperty(meta::Metadata, name::Symbol)
    if name == :df
        return getfield(meta, name)
    else
        meta.df[1, name]
    end
end

# For `meta.name = value` syntax
function Base.setproperty!(meta::Metadata, name::Symbol, value)
    if name == :df
        setfield!(meta, name, value)
    else
        meta.df[1, name] = value
    end
end

mutable struct Data
    df::DataFrame
    metadata::Metadata
    name::String
    savedindex::Integer
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
    metadata = Metadata(DataFrame(
        [[v"0.0.0"], [0],[column_names]],
        [:version, :num_samples, :column_names]
    ))
    return Data(df, metadata, name, 0)
end

Base.push!(data::Data, val) = push!(data.df, val)

struct Experiment
    device::Device
    raw_data::Data
    tags::Array{String,1}
    extra_info::Dict{Symbol,Any}
    folderpath::String
    # cases::Array{Symbol,1}
end

# For `experiment.num_samples`
function Base.getproperty(experiment::Experiment, name::Symbol)
    if name == :num_samples
        return experiment.raw_data.metadata.num_samples
    else
        return getfield(experiment, name)
    end
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

"""
    clear!(experiment::Experiment)

Delete all data from experiment.
"""
function clear!(experiment::Experiment)
    d = experiment.raw_data
    empty!(d.df)
    d.metadata.num_samples = 0
    d.savedindex = 0
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

Gather raw EEG data. 

`runtime`: Runtime in seconds.

Optional arguments:

`tags`: Tags to add to every data point on top of tags given to [`Experiment`](@ref)

`extra_info`: Extra info to add to every data point on top of extra info given to [`Experiment`](@ref)



TODO: create test
"""
function gather_data!(experiment::Experiment, runtime::Number; tags::Array=[], extra_info::Dict=Dict(),
    delay::Number=0, save::Bool=true)

    start_time = time()
    all_tags = vcat(experiment.tags, string.(tags))
    new_row = Array{Any,1}(undef, 3 + experiment.device.board.num_channels)
    new_row[2] = all_tags
    new_row[3] = merge(experiment.extra_info, extra_info)
    new_num = 0
    if !iscompatible(experiment.raw_data, load_metadata(experiment))
        throw(ErrorException("TODO")) # TODO
    end
    while (time() - start_time) < runtime
        new_row[1] = time()
        for channel = 1:experiment.device.board.num_channels
            new_row[3+channel] = get_voltage(experiment.device.board, channel)
        end
        push!(experiment.raw_data, new_row)
        if save
            save_data(experiment, updatemeta=false, checkmeta=false)
        end
        if delay != 0
            sleep(delay)
        end
        new_num += 1
    end
    experiment.raw_data.metadata.num_samples += new_num
end

get_metadatapath(folder_path, name) = joinpath(folder_path, name * "Metadata.csv")

function load_metadata(path::String)
    if isfile(path)
        df = CSV.read(path, DataFrame)
        return Metadata(df)
    else
        return nothing
    end
end

load_metadata(folder_path, name) = load_metadata(get_metadatapath(folder_path, name))
load_metadata(experiment::Experiment) = load_metadata(experiment.folderpath, "RawData")

"""
    is_compat(df::DataFrame, metadata::Union{Metadata, Nothing})

Check if `df` is "compatible" with `metadata`.

If `metadata` is `nothing`, then return true (for easy use with `load_metadata`
which returns nothing when no `metadata` is defined)
"""
function iscompatible(df::DataFrame, metadata::Union{Metadata,Nothing})
    if metadata === nothing
        return true
    end
    if metadata.column_names == names(df)
        return true
    end
    return false
end

iscompatible(data::Data, metadata::Union{Metadata,Nothing}) = iscompatible(
    data.df,
    metadata
)

function combine_metadata(data::Data, meta::Metadata; repeat=false)
    new_meta = copy(meta)
    add_length = data.metadata.num_samples
    if !repeat
        add_length -= data.savedindex
    end
    new_meta.metadata.num_samples += add_length
    return new_meta
end

"""
    create_path(path::String)

Create necessary folders and file if they don't exist yet, so that
`isdir(path)` or `isfile(path)` returns true depending on whether
`path` points to a folder or a file.
"""
function createpath(path::String)
    mkpath(dirname(path)) # create directories containing file
    if isfile(path)
        io = open(path, create=true) # create file
        close(io)
    end
end

function save_data(data::Data, df::DataFrame, folderpath; checkmeta=true, updatemeta=true, repeat=false)
    datapath = joinpath(folderpath, data.name * ".csv")
    metapath = get_metadatapath(folderpath, data.name)
    metadata = nothing
    if checkmeta && isfile(metapath)
        if !is_compat(data.df, get_metadata(metapath))
            throw(ErrorException( # TODO: more specific error
                "Data not compatible with DataFrame at $file_path 
                (perhaps processed vs non-processed data or different 
                number of channels). If you want to overwrite the old data,
                delete both $file_path and $meta_file_path."))
        end
    end
    if updatemeta
        if metadata === nothing
            metadata = load_metadata(metapath)
        end
        if metadata === nothing
            newmeta = data.metadata
        else
            newmeta = combine_metadata(data, metadata)
        end
        CSV.write(metapath, newmeta.df)
    end
    # Make sure file exist
    overwrite = false
    if !isfile(datapath)
        createpath(datapath)
        overwrite = true # overwrite to create headers
    end
    # deactivate append if overwrite --> file is overwritten
    CSV.write(datapath, df, append=!overwrite)
end

function save_data(data::Data, folderpath; checkmeta=true, updatemeta=true, repeat=false)
    if !isempty(data.df)
        if repeat
            df_new = data.df
        else
            df_new = data.df[(data.savedindex+1):end, :]
        end
        save_data(data, df_new, folderpath, checkmeta=checkmeta, updatemeta=updatemeta, repeat=repeat)
        data.savedindex = data.metadata.num_samples
    end
end

function save_data(experiment::Experiment; checkmeta=true, updatemeta=true, repeat=false)
    save_data(experiment.raw_data, experiment.folderpath, checkmeta=checkmeta,
        updatemeta=updatemeta, repeat=repeat)
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